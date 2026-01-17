using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Threading;
using SpeedReaderApp.Models;
using SpeedReaderApp.Services;

namespace SpeedReaderApp.ViewModels;

public class MainWindowViewModel : INotifyPropertyChanged, IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly DatabaseService _database;
    private readonly PdfService _pdfService = new();
    private Timer? _timer;
    private bool _disposed;
    private string[] _words = Array.Empty<string>();
    private int _currentWordIndex = 0;
    private bool _isPlaying = false;
    private int _wpm = 300; // Default 300 WPM (optimal range)
    private string _inputText = "Welcome to Speed Reader! This app uses RSVP (Rapid Serial Visual Presentation) with ORP (Optimal Recognition Point) highlighting. Paste your text here and click Start to begin reading. Science shows best comprehension at 250-350 WPM, but you can go up to 2000 WPM!";
    private string _currentWord = "";
    private string _orpHighlightedWord = "";
    private string _textColor = "#FFFFFF";
    private string _backgroundColor = "#1E1E1E";
    private string _orpColor = "#FF4444";
    private string _definitionText = "";
    private bool _showDefinition = false;
    private int _progress = 0;
    private string _pausedWord = "";
    private bool _isLoading = false;
    private string _loadingMessage = "";
    private bool _showLibrary = false;
    private SavedBook? _currentBook = null;

    public event PropertyChangedEventHandler? PropertyChanged;

    public MainWindowViewModel()
    {
        // Configure HttpClient with timeout
        _httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromSeconds(10)
        };

        _database = new DatabaseService(DatabaseService.GetDatabasePath());
        SavedBooks = new ObservableCollection<SavedBook>();
        _ = LoadBooksAsync();
    }

    public void Dispose()
    {
        if (_disposed) return;

        _timer?.Dispose();
        _httpClient?.Dispose();
        _disposed = true;
    }

    public ObservableCollection<SavedBook> SavedBooks { get; }

    public string InputText
    {
        get => _inputText;
        set
        {
            _inputText = value;
            OnPropertyChanged();
            ResetReader();
        }
    }

    public string CurrentWord
    {
        get => _currentWord;
        private set
        {
            _currentWord = value;
            OnPropertyChanged();
            UpdateORPHighlightedWord();
        }
    }

    public string ORPHighlightedWord
    {
        get => _orpHighlightedWord;
        private set
        {
            _orpHighlightedWord = value;
            OnPropertyChanged();
        }
    }

    public int WPM
    {
        get => _wpm;
        set
        {
            if (value < 50) value = 50;
            if (value > 2000) value = 2000;
            _wpm = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(IntervalMs));
            if (_isPlaying)
            {
                RestartTimer();
            }
        }
    }

    public int IntervalMs => 60000 / _wpm;

    public bool IsPlaying
    {
        get => _isPlaying;
        private set
        {
            _isPlaying = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(PlayPauseText));
        }
    }

    public string PlayPauseText => IsPlaying ? "⏸ Pause" : "▶ Play";

    public string TextColor
    {
        get => _textColor;
        set
        {
            _textColor = value;
            OnPropertyChanged();
        }
    }

    public string BackgroundColor
    {
        get => _backgroundColor;
        set
        {
            _backgroundColor = value;
            OnPropertyChanged();
        }
    }

    public string ORPColor
    {
        get => _orpColor;
        set
        {
            _orpColor = value;
            OnPropertyChanged();
        }
    }

    public int Progress
    {
        get => _progress;
        private set
        {
            _progress = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(ProgressText));
        }
    }

    public string ProgressText => _words.Length > 0 ? $"{_currentWordIndex + 1} / {_words.Length}" : "0 / 0";

    public string DefinitionText
    {
        get => _definitionText;
        private set
        {
            _definitionText = value;
            OnPropertyChanged();
        }
    }

    public bool ShowDefinition
    {
        get => _showDefinition;
        private set
        {
            _showDefinition = value;
            OnPropertyChanged();
        }
    }

    public void TogglePlayPause()
    {
        if (IsPlaying)
        {
            Pause();
        }
        else
        {
            Play();
        }
    }

    public void Play()
    {
        if (_words.Length == 0)
        {
            ResetReader();
        }

        if (_currentWordIndex >= _words.Length)
        {
            _currentWordIndex = 0;
        }

        IsPlaying = true;
        ShowDefinition = false;
        StartTimer();
    }

    public void Pause()
    {
        IsPlaying = false;
        _timer?.Dispose();
        _timer = null;
        _pausedWord = CurrentWord;
    }

    public void Reset()
    {
        Pause();
        ResetReader();
    }

    public void PreviousWord()
    {
        if (_currentWordIndex > 0)
        {
            _currentWordIndex--;
            DisplayCurrentWord();
        }
    }

    public void NextWord()
    {
        if (_currentWordIndex < _words.Length - 1)
        {
            _currentWordIndex++;
            DisplayCurrentWord();
        }
    }

    public void IncreaseSpeed()
    {
        // Increase by 50 WPM
        WPM += 50;
    }

    public void DecreaseSpeed()
    {
        // Decrease by 50 WPM
        WPM -= 50;
    }

    public void SetSpeedPreset(int wpm)
    {
        WPM = wpm;
    }

    public async Task LookupDefinition()
    {
        if (string.IsNullOrWhiteSpace(_pausedWord))
        {
            _pausedWord = CurrentWord;
        }

        // Remove punctuation and trim
        var word = new string(_pausedWord.Where(char.IsLetter).ToArray()).Trim().ToLower();
        if (string.IsNullOrWhiteSpace(word))
        {
            DefinitionText = "No word selected";
            ShowDefinition = true;
            return;
        }

        try
        {
            DefinitionText = "Loading definition...";
            ShowDefinition = true;

            var response = await _httpClient.GetStringAsync($"https://api.dictionaryapi.dev/api/v2/entries/en/{word}");
            var jsonDoc = JsonDocument.Parse(response);

            if (jsonDoc.RootElement.GetArrayLength() > 0)
            {
                var firstEntry = jsonDoc.RootElement[0];
                var wordText = firstEntry.GetProperty("word").GetString() ?? word;
                var phonetic = firstEntry.TryGetProperty("phonetic", out var phoneticEl)
                    ? phoneticEl.GetString() ?? ""
                    : "";

                var definitions = new List<string>();

                if (firstEntry.TryGetProperty("meanings", out var meanings))
                {
                    foreach (var meaning in meanings.EnumerateArray())
                    {
                        var partOfSpeech = meaning.GetProperty("partOfSpeech").GetString() ?? "";

                        if (meaning.TryGetProperty("definitions", out var defs))
                        {
                            var defList = defs.EnumerateArray().Take(2).ToList();
                            for (int i = 0; i < defList.Count; i++)
                            {
                                var def = defList[i].GetProperty("definition").GetString() ?? "";
                                definitions.Add($"{partOfSpeech}: {def}");
                            }
                        }
                    }
                }

                var result = $"📖 {wordText}";
                if (!string.IsNullOrEmpty(phonetic))
                {
                    result += $" {phonetic}";
                }
                result += "\n\n" + string.Join("\n\n", definitions.Take(3));

                DefinitionText = result;
            }
            else
            {
                DefinitionText = $"No definition found for '{word}'";
            }
        }
        catch (HttpRequestException)
        {
            DefinitionText = $"Could not find definition for '{word}'.\n\nMake sure you're connected to the internet.";
        }
        catch (Exception ex)
        {
            DefinitionText = $"Error looking up '{word}': {ex.Message}";
        }

        ShowDefinition = true;
    }

    public void CloseDefinition()
    {
        ShowDefinition = false;
    }

    public bool IsLoading
    {
        get => _isLoading;
        private set
        {
            _isLoading = value;
            OnPropertyChanged();
        }
    }

    public string LoadingMessage
    {
        get => _loadingMessage;
        private set
        {
            _loadingMessage = value;
            OnPropertyChanged();
        }
    }

    public bool ShowLibrary
    {
        get => _showLibrary;
        set
        {
            _showLibrary = value;
            OnPropertyChanged();
        }
    }

    public async Task LoadPdfAsync(Stream pdfStream, string fileName)
    {
        try
        {
            IsLoading = true;
            LoadingMessage = "Reading PDF...";

            var (content, title) = await _pdfService.ExtractTextFromPdfAsync(pdfStream);

            if (string.IsNullOrWhiteSpace(content))
            {
                LoadingMessage = "Error: PDF appears to be empty or contains only images.";
                await Task.Delay(3000);
                IsLoading = false;
                return;
            }

            LoadingMessage = $"Processing {content.Split(' ').Length} words...";

            // Use filename as fallback title
            if (title == "Untitled Document" && !string.IsNullOrWhiteSpace(fileName))
            {
                title = Path.GetFileNameWithoutExtension(fileName);
            }

            // Save to database
            var book = new SavedBook
            {
                Title = title,
                Content = content,
                Source = "PDF",
                TotalWords = content.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries).Length,
                ReadingSpeed = WPM
            };

            await _database.SaveBookAsync(book);
            await LoadBooksAsync();

            // Load the book
            _currentBook = book;
            InputText = content;

            LoadingMessage = "PDF loaded successfully!";
            await Task.Delay(1000);
            IsLoading = false;
        }
        catch (Exception ex)
        {
            LoadingMessage = $"Error loading PDF: {ex.Message}";
            await Task.Delay(3000);
            IsLoading = false;
        }
    }

    public async Task LoadBookAsync(SavedBook book)
    {
        _currentBook = book;
        InputText = book.Content;
        WPM = book.ReadingSpeed > 0 ? book.ReadingSpeed : 300;

        // Resume from last position
        if (book.LastPosition > 0 && book.LastPosition < _words.Length)
        {
            _currentWordIndex = book.LastPosition;
            DisplayCurrentWord();
        }

        ShowLibrary = false;

        // Update last read time
        await _database.UpdateReadingProgressAsync(book.Id, book.LastPosition, WPM);
    }

    public async Task SaveCurrentBookAsync(string title)
    {
        if (string.IsNullOrWhiteSpace(_inputText))
            return;

        var book = new SavedBook
        {
            Title = title,
            Content = _inputText,
            Source = "Manual",
            TotalWords = _words.Length,
            ReadingSpeed = WPM,
            LastPosition = _currentWordIndex
        };

        await _database.SaveBookAsync(book);
        await LoadBooksAsync();
        _currentBook = book;
    }

    public async Task DeleteBookAsync(SavedBook book)
    {
        await _database.DeleteBookAsync(book);
        await LoadBooksAsync();

        if (_currentBook?.Id == book.Id)
        {
            _currentBook = null;
        }
    }

    public async Task UpdateCurrentProgress()
    {
        if (_currentBook != null && _currentBook.Id != 0)
        {
            await _database.UpdateReadingProgressAsync(_currentBook.Id, _currentWordIndex, WPM);
        }
    }

    public void ToggleLibrary()
    {
        ShowLibrary = !ShowLibrary;
    }

    private async Task LoadBooksAsync()
    {
        var books = await _database.GetBooksAsync();
        SavedBooks.Clear();
        foreach (var book in books)
        {
            SavedBooks.Add(book);
        }
    }

    private void ResetReader()
    {
        _words = SplitIntoWords(_inputText);
        _currentWordIndex = 0;
        Progress = 0;
        if (_words.Length > 0)
        {
            DisplayCurrentWord();
        }
        else
        {
            CurrentWord = "";
        }
    }

    private void StartTimer()
    {
        _timer?.Dispose();
        _timer = new Timer(_ => AdvanceWord(), null, IntervalMs, IntervalMs);
        DisplayCurrentWord();
    }

    private void RestartTimer()
    {
        if (_isPlaying)
        {
            StartTimer();
        }
    }

    private void AdvanceWord()
    {
        // Marshal to UI thread to avoid cross-thread UI updates
        Dispatcher.UIThread.Post(() =>
        {
            if (_currentWordIndex < _words.Length)
            {
                DisplayCurrentWord();
                _currentWordIndex++;
                Progress = _words.Length > 0 ? (_currentWordIndex * 100 / _words.Length) : 0;
            }
            else
            {
                IsPlaying = false;
                _timer?.Dispose();
                _timer = null;
            }
        });
    }

    private void DisplayCurrentWord()
    {
        if (_currentWordIndex < _words.Length)
        {
            CurrentWord = _words[_currentWordIndex];
        }
    }

    private void UpdateORPHighlightedWord()
    {
        if (string.IsNullOrEmpty(CurrentWord))
        {
            ORPHighlightedWord = "";
            return;
        }

        // Calculate ORP (Optimal Recognition Point)
        // For words, the ORP is typically slightly left of center
        var word = CurrentWord;
        int orpIndex = CalculateORPIndex(word.Length);

        // Build the highlighted word with inline color formatting
        // We'll use a special format that the view can parse
        ORPHighlightedWord = $"{word.Substring(0, orpIndex)}|{word[orpIndex]}|{word.Substring(orpIndex + 1)}";
    }

    private int CalculateORPIndex(int wordLength)
    {
        // ORP formula: approximately 1/3 to 1/2 from the start
        // Research shows the ORP is slightly left of center
        if (wordLength <= 1) return 0;
        if (wordLength <= 3) return 1;
        if (wordLength <= 5) return 2;
        if (wordLength <= 8) return wordLength / 2;
        return (wordLength / 2) - 1;
    }

    private string[] SplitIntoWords(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return Array.Empty<string>();

        return text.Split(new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);
    }

    protected void OnPropertyChanged([CallerMemberName] string? propertyName = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}
