use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};
use std::thread;

use anyhow::{Context, Result, anyhow};
use tracing::Level;

use crate::logging::{init_logging_with_filter, log_event};
use crate::provider::clear_registered_caches;

const DEFAULT_FREQ: &str = "__DEFAULT_FREQ";

/// Region identifiers mirroring qlib's constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Region {
    China,
    UnitedStates,
    Taiwan,
}

impl Region {
    pub fn as_str(self) -> &'static str {
        match self {
            Region::China => "cn",
            Region::UnitedStates => "us",
            Region::Taiwan => "tw",
        }
    }
}

/// Default configuration templates that mirror qlib's `client` and `server` modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DefaultConfig {
    #[default]
    Client,
    Server,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoggingSettings {
    pub level: Level,
    pub filter: Option<String>,
}

impl Default for LoggingSettings {
    fn default() -> Self {
        Self {
            level: Level::INFO,
            filter: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigState {
    mode: DefaultConfig,
    region: Region,
    provider_uri: HashMap<String, PathBuf>,
    mount_path: HashMap<String, PathBuf>,
    auto_mount: bool,
    logging: LoggingSettings,
    expression_cache: Option<String>,
    dataset_cache: Option<String>,
    kernels: usize,
}

impl ConfigState {
    fn data_path_manager(&self) -> DataPathManager {
        DataPathManager {
            provider_uri: self.provider_uri.clone(),
            mount_path: self.mount_path.clone(),
        }
    }
}

#[derive(Clone, Debug)]
struct ConfigTemplate {
    provider_uri: HashMap<String, PathBuf>,
    mount_path: HashMap<String, PathBuf>,
    auto_mount: bool,
    logging: LoggingSettings,
    expression_cache: Option<String>,
    dataset_cache: Option<String>,
    kernels: usize,
}

impl ConfigTemplate {
    fn new_default(provider: &str, auto_mount: bool) -> Self {
        let provider_path = PathBuf::from(provider);
        let mut provider_uri = HashMap::new();
        provider_uri.insert(DEFAULT_FREQ.to_string(), provider_path.clone());

        let mut mount_path = HashMap::new();
        mount_path.insert(DEFAULT_FREQ.to_string(), provider_path.clone());

        Self {
            provider_uri,
            mount_path,
            auto_mount,
            logging: LoggingSettings::default(),
            expression_cache: None,
            dataset_cache: None,
            kernels: compute_default_kernels(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConfigSnapshot {
    pub mode: DefaultConfig,
    pub region: Region,
    pub provider_uri: HashMap<String, PathBuf>,
    pub mount_path: HashMap<String, PathBuf>,
    pub auto_mount: bool,
    pub logging: LoggingSettings,
    pub expression_cache: Option<String>,
    pub dataset_cache: Option<String>,
    pub kernels: usize,
    pub registered: bool,
}

#[derive(Debug)]
pub struct QlibConfig {
    state: ConfigState,
    client_template: ConfigTemplate,
    server_template: ConfigTemplate,
    registered: bool,
}

impl QlibConfig {
    fn new() -> Self {
        let client_template = ConfigTemplate::new_default("~/.qlib/qlib_data/cn_data", false);
        let server_template = ConfigTemplate::new_default("~/.qlib/qlib_data/cn_data", true);

        Self {
            state: ConfigState {
                mode: DefaultConfig::Client,
                region: Region::China,
                provider_uri: client_template.provider_uri.clone(),
                mount_path: client_template.mount_path.clone(),
                auto_mount: client_template.auto_mount,
                logging: client_template.logging.clone(),
                expression_cache: client_template.expression_cache.clone(),
                dataset_cache: client_template.dataset_cache.clone(),
                kernels: client_template.kernels,
            },
            client_template,
            server_template,
            registered: false,
        }
    }

    fn set(&mut self, mode: DefaultConfig, options: &InitOptions) -> Result<()> {
        self.registered = false;
        self.apply_mode(mode);
        self.apply_region(options.region.unwrap_or(self.state.region));
        self.apply_overrides(options)?;
        self.resolve_paths()?;
        Ok(())
    }

    fn apply_mode(&mut self, mode: DefaultConfig) {
        self.state.mode = mode;
        let template = match mode {
            DefaultConfig::Client => &self.client_template,
            DefaultConfig::Server => &self.server_template,
        };

        self.state.provider_uri = template.provider_uri.clone();
        self.state.mount_path = template.mount_path.clone();
        self.state.auto_mount = template.auto_mount;
        self.state.logging = template.logging.clone();
        self.state.expression_cache = template.expression_cache.clone();
        self.state.dataset_cache = template.dataset_cache.clone();
        self.state.kernels = template.kernels;
    }

    fn apply_region(&mut self, region: Region) {
        self.state.region = region;
    }

    fn apply_overrides(&mut self, options: &InitOptions) -> Result<()> {
        if let Some(ref provider_uri) = options.provider_uri {
            self.state.provider_uri = normalize_provider_map(provider_uri);
        }

        if let Some(ref mount_path) = options.mount_path {
            self.state.mount_path = normalize_mount_map(mount_path);
        }

        if let Some(auto_mount) = options.auto_mount {
            self.state.auto_mount = auto_mount;
        }

        if let Some(level) = options.logging_level {
            self.state.logging.level = level;
        }

        if let Some(ref filter) = options.logging_filter {
            self.state.logging.filter = Some(filter.clone());
        }

        if let Some(region) = options.region {
            self.apply_region(region);
        }

        if let Some(value) = &options.expression_cache {
            self.state.expression_cache = value.clone();
        }

        if let Some(value) = &options.dataset_cache {
            self.state.dataset_cache = value.clone();
        }

        if let Some(kernels) = options.kernels {
            self.state.kernels = kernels.max(1);
        }

        Ok(())
    }

    fn resolve_paths(&mut self) -> Result<()> {
        self.state.provider_uri = self
            .state
            .provider_uri
            .iter()
            .map(|(freq, path)| (freq.clone(), expand_user(path)))
            .collect();

        let mut resolved_mount = HashMap::new();
        if self.state.mount_path.is_empty() {
            for (freq, path) in &self.state.provider_uri {
                resolved_mount.insert(freq.clone(), path.clone());
            }
        } else {
            for (freq, path) in &self.state.mount_path {
                resolved_mount.insert(freq.clone(), expand_user(path));
            }
        }

        for freq in self.state.provider_uri.keys() {
            resolved_mount
                .entry(freq.clone())
                .or_insert_with(|| self.state.provider_uri[freq].clone());
        }

        self.state.mount_path = resolved_mount;

        for uri in self.state.provider_uri.values() {
            let uri_type = DataPathManager::get_uri_type(uri);
            if matches!(uri_type, UriType::Local) && !uri.exists() && self.state.auto_mount {
                std::fs::create_dir_all(uri).with_context(|| {
                    format!("failed to create provider uri directory {}", uri.display())
                })?;
            }
        }

        Ok(())
    }

    fn register(&mut self) {
        self.registered = true;
        log_event(
            file!(),
            "QlibConfig",
            "register",
            "config",
            line!(),
            &format!(
                "qlib initialized in {:?} mode for region {}",
                self.state.mode,
                self.state.region.as_str()
            ),
            None,
            "post",
            "POST",
        );
    }

    fn logging(&self) -> &LoggingSettings {
        &self.state.logging
    }

    fn snapshot(&self) -> ConfigSnapshot {
        ConfigSnapshot {
            mode: self.state.mode,
            region: self.state.region,
            provider_uri: self.state.provider_uri.clone(),
            mount_path: self.state.mount_path.clone(),
            auto_mount: self.state.auto_mount,
            logging: self.state.logging.clone(),
            expression_cache: self.state.expression_cache.clone(),
            dataset_cache: self.state.dataset_cache.clone(),
            kernels: self.state.kernels,
            registered: self.registered,
        }
    }

    fn registered(&self) -> bool {
        self.registered
    }

    fn data_path_manager(&self) -> DataPathManager {
        self.state.data_path_manager()
    }

    fn provider_entries(&self) -> Vec<(String, PathBuf, UriType, PathBuf)> {
        self.state
            .provider_uri
            .iter()
            .map(|(freq, uri)| {
                let uri_type = DataPathManager::get_uri_type(uri);
                let mount = self
                    .state
                    .mount_path
                    .get(freq)
                    .cloned()
                    .unwrap_or_else(|| uri.clone());
                (freq.clone(), uri.clone(), uri_type, mount)
            })
            .collect()
    }

    fn auto_mount(&self) -> bool {
        self.state.auto_mount
    }
}

fn compute_default_kernels() -> usize {
    let available = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    available.saturating_sub(2).max(1)
}

fn normalize_provider_map(map: &HashMap<String, PathBuf>) -> HashMap<String, PathBuf> {
    if map.is_empty() {
        return HashMap::from([(DEFAULT_FREQ.to_string(), PathBuf::from("."))]);
    }

    let mut normalized = HashMap::new();
    for (freq, path) in map {
        normalized.insert(freq.clone(), path.clone());
    }

    if !normalized.contains_key(DEFAULT_FREQ)
        && let Some((_freq, path)) = normalized.iter().next()
    {
        normalized.insert(DEFAULT_FREQ.to_string(), path.clone());
    }

    normalized
}

fn normalize_mount_map(map: &HashMap<String, PathBuf>) -> HashMap<String, PathBuf> {
    if map.is_empty() {
        return HashMap::new();
    }

    let mut normalized = HashMap::new();
    for (freq, path) in map {
        normalized.insert(freq.clone(), path.clone());
    }

    normalized
}

fn expand_user(path: &Path) -> PathBuf {
    let path_str = path.to_string_lossy();
    if path_str == "~" {
        return home_dir().unwrap_or_else(|| PathBuf::from("~"));
    }

    if let Some(stripped) = path_str.strip_prefix("~/")
        && let Some(home) = home_dir()
    {
        return home.join(stripped);
    }

    PathBuf::from(path)
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

#[derive(Clone, Debug)]
pub struct DataPathManager {
    provider_uri: HashMap<String, PathBuf>,
    mount_path: HashMap<String, PathBuf>,
}

impl DataPathManager {
    pub fn get_data_uri(&self, freq: Option<&str>) -> PathBuf {
        let target_freq = freq.unwrap_or(DEFAULT_FREQ);
        if let Some(path) = self.provider_uri.get(target_freq) {
            return path.clone();
        }
        self.provider_uri
            .get(DEFAULT_FREQ)
            .cloned()
            .unwrap_or_else(|| PathBuf::from("."))
    }

    fn get_uri_type(path: &Path) -> UriType {
        let display = path.to_string_lossy();
        if is_windows_path(&display) {
            return UriType::Local;
        }

        if display.contains(':') {
            return UriType::Nfs;
        }

        UriType::Local
    }

    pub fn mount_path(&self, freq: &str) -> Option<PathBuf> {
        self.mount_path.get(freq).cloned()
    }
}

#[derive(Clone, Copy, Debug)]
enum UriType {
    Local,
    Nfs,
}

fn is_windows_path(path: &str) -> bool {
    let bytes = path.as_bytes();
    if bytes.len() < 2 {
        return false;
    }
    bytes[1] == b':' && bytes[0].is_ascii_alphabetic()
}

#[derive(Clone, Debug)]
pub struct InitOptions {
    pub provider_uri: Option<HashMap<String, PathBuf>>,
    pub mount_path: Option<HashMap<String, PathBuf>>,
    pub auto_mount: Option<bool>,
    pub logging_level: Option<Level>,
    pub logging_filter: Option<String>,
    pub region: Option<Region>,
    pub expression_cache: Option<Option<String>>,
    pub dataset_cache: Option<Option<String>>,
    pub kernels: Option<usize>,
    pub skip_if_registered: bool,
    pub clear_mem_cache: bool,
}

impl InitOptions {
    pub fn without_cache_clear() -> Self {
        Self {
            clear_mem_cache: false,
            ..Self::default()
        }
    }
}

impl Default for InitOptions {
    fn default() -> Self {
        Self {
            provider_uri: None,
            mount_path: None,
            auto_mount: None,
            logging_level: None,
            logging_filter: None,
            region: None,
            expression_cache: None,
            dataset_cache: None,
            kernels: None,
            skip_if_registered: false,
            clear_mem_cache: true,
        }
    }
}

static GLOBAL_CONFIG: OnceLock<RwLock<QlibConfig>> = OnceLock::new();

fn config_lock() -> &'static RwLock<QlibConfig> {
    GLOBAL_CONFIG.get_or_init(|| RwLock::new(QlibConfig::new()))
}

pub fn init(mode: DefaultConfig, options: InitOptions) -> Result<()> {
    let lock = config_lock();
    let mut guard = lock
        .write()
        .map_err(|_| anyhow!("qlib config lock poisoned"))?;

    if options.skip_if_registered && guard.registered() {
        log_event(
            file!(),
            "QlibConfig",
            "init_skip",
            "config",
            line!(),
            "Initialization skipped because configuration is already registered",
            None,
            "none",
            "GET",
        );
        return Ok(());
    }

    if options.clear_mem_cache {
        clear_registered_caches().map_err(|err| anyhow!(err))?;
    }

    guard.set(mode, &options)?;
    let logging = guard.logging().clone();
    let provider_entries = guard.provider_entries();
    let auto_mount = guard.auto_mount();
    init_logging_with_filter(logging.filter.as_deref(), Some(logging.level))?;
    for (freq, uri, uri_type, mount) in provider_entries {
        match uri_type {
            UriType::Local => {
                let message = if uri.exists() {
                    format!("Provider URI for {freq} resolved to {}", uri.display())
                } else if auto_mount {
                    format!(
                        "Provider URI for {freq} created at {} due to auto_mount",
                        uri.display()
                    )
                } else {
                    format!(
                        "Provider URI for {freq} missing at {} (auto_mount disabled)",
                        uri.display()
                    )
                };
                log_event(
                    file!(),
                    "QlibConfig",
                    "provider_uri",
                    "config.provider",
                    line!(),
                    &message,
                    None,
                    "none",
                    "GET",
                );
            }
            UriType::Nfs => {
                log_event(
                    file!(),
                    "QlibConfig",
                    "provider_uri",
                    "config.provider",
                    line!(),
                    &format!(
                        "NFS provider URI for {freq} resolved to {} with mount {}",
                        uri.display(),
                        mount.display()
                    ),
                    None,
                    "none",
                    "GET",
                );
            }
        }
    }
    guard.register();

    Ok(())
}

pub fn config_snapshot() -> Option<ConfigSnapshot> {
    let lock = GLOBAL_CONFIG.get()?;
    let guard = lock.read().ok()?;
    Some(guard.snapshot())
}

pub fn with_data_path<F, R>(func: F) -> Result<R>
where
    F: FnOnce(&DataPathManager) -> R,
{
    let lock = config_lock();
    let guard = lock
        .read()
        .map_err(|_| anyhow!("qlib config lock poisoned"))?;
    Ok(func(&guard.data_path_manager()))
}

pub const REG_CN: Region = Region::China;
pub const REG_US: Region = Region::UnitedStates;
pub const REG_TW: Region = Region::Taiwan;

pub fn reset_for_tests() {
    if let Some(lock) = GLOBAL_CONFIG.get()
        && let Ok(mut guard) = lock.write()
    {
        *guard = QlibConfig::new();
    }
}
