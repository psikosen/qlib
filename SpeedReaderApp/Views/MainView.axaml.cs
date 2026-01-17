using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using SpeedReaderApp.ViewModels;

namespace SpeedReaderApp.Views;

public partial class MainView : UserControl
{
    private MainWindowViewModel ViewModel => (MainWindowViewModel)DataContext!;

    public MainView()
    {
        InitializeComponent();
    }

    private void TogglePlayPause(object? sender, RoutedEventArgs e)
    {
        ViewModel.TogglePlayPause();
    }

    private void Reset(object? sender, RoutedEventArgs e)
    {
        ViewModel.Reset();
    }

    private void PreviousWord(object? sender, RoutedEventArgs e)
    {
        ViewModel.PreviousWord();
    }

    private void NextWord(object? sender, RoutedEventArgs e)
    {
        ViewModel.NextWord();
    }

    private void IncreaseSpeed(object? sender, RoutedEventArgs e)
    {
        ViewModel.IncreaseSpeed();
    }

    private void DecreaseSpeed(object? sender, RoutedEventArgs e)
    {
        ViewModel.DecreaseSpeed();
    }

    private void SetSpeed250(object? sender, RoutedEventArgs e)
    {
        ViewModel.SetSpeedPreset(250);
    }

    private void SetSpeed300(object? sender, RoutedEventArgs e)
    {
        ViewModel.SetSpeedPreset(300);
    }

    private void SetSpeed400(object? sender, RoutedEventArgs e)
    {
        ViewModel.SetSpeedPreset(400);
    }

    private void SetSpeed500(object? sender, RoutedEventArgs e)
    {
        ViewModel.SetSpeedPreset(500);
    }

    private void SetSpeed1000(object? sender, RoutedEventArgs e)
    {
        ViewModel.SetSpeedPreset(1000);
    }

    private async void LookupDefinition(object? sender, RoutedEventArgs e)
    {
        await ViewModel.LookupDefinition();
    }

    private void CloseDefinition(object? sender, RoutedEventArgs e)
    {
        ViewModel.CloseDefinition();
    }

    private void SetLightTheme(object? sender, RoutedEventArgs e)
    {
        ViewModel.BackgroundColor = "#FFFFFF";
        ViewModel.TextColor = "#000000";
        ViewModel.ORPColor = "#FF0000";
    }

    private void SetDarkTheme(object? sender, RoutedEventArgs e)
    {
        ViewModel.BackgroundColor = "#1E1E1E";
        ViewModel.TextColor = "#FFFFFF";
        ViewModel.ORPColor = "#FF4444";
    }

    private void SetSepiaTheme(object? sender, RoutedEventArgs e)
    {
        ViewModel.BackgroundColor = "#F4ECD8";
        ViewModel.TextColor = "#5C4B37";
        ViewModel.ORPColor = "#8B0000";
    }

    private void SetHighContrastTheme(object? sender, RoutedEventArgs e)
    {
        ViewModel.BackgroundColor = "#000000";
        ViewModel.TextColor = "#FFFF00";
        ViewModel.ORPColor = "#00FF00";
    }

    // Single tap toggles play/pause
    private void OnWordTapped(object? sender, TappedEventArgs e)
    {
        if (!ViewModel.ShowDefinition)
        {
            ViewModel.TogglePlayPause();
        }
    }

    // Double tap looks up definition (when paused)
    private async void OnWordDoubleTapped(object? sender, TappedEventArgs e)
    {
        if (!ViewModel.IsPlaying)
        {
            await ViewModel.LookupDefinition();
        }
    }
}
