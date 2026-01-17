using SpeedReaderApp.ViewModels;
using Xunit;

namespace SpeedReaderApp.Tests.ViewModels;

public class MainWindowViewModelTests
{
    [Fact]
    public void Constructor_InitializesWithDefaultValues()
    {
        // Arrange & Act
        var viewModel = new MainWindowViewModel();

        // Assert
        Assert.Equal(300, viewModel.WPM);
        Assert.False(viewModel.IsPlaying);
        Assert.NotEmpty(viewModel.InputText);
        Assert.Equal("#1E1E1E", viewModel.BackgroundColor);
        Assert.Equal("#FFFFFF", viewModel.TextColor);
        Assert.False(viewModel.ShowDefinition);
        Assert.False(viewModel.IsLoading);
        Assert.False(viewModel.ShowLibrary);
    }

    [Theory]
    [InlineData(250)]
    [InlineData(300)]
    [InlineData(500)]
    [InlineData(1000)]
    [InlineData(2000)]
    public void SetSpeedPreset_SetsCorrectWPM(int wpm)
    {
        // Arrange
        var viewModel = new MainWindowViewModel();

        // Act
        viewModel.SetSpeedPreset(wpm);

        // Assert
        Assert.Equal(wpm, viewModel.WPM);
    }

    [Fact]
    public void WPM_ClampsToBounds()
    {
        // Arrange
        var viewModel = new MainWindowViewModel();

        // Act & Assert - Lower bound
        viewModel.WPM = 10;
        Assert.Equal(50, viewModel.WPM);

        // Act & Assert - Upper bound
        viewModel.WPM = 3000;
        Assert.Equal(2000, viewModel.WPM);

        // Act & Assert - Valid value
        viewModel.WPM = 500;
        Assert.Equal(500, viewModel.WPM);
    }

    [Theory]
    [InlineData(60, 1000)]   // 60 WPM = 1000ms per word
    [InlineData(300, 200)]   // 300 WPM = 200ms per word
    [InlineData(600, 100)]   // 600 WPM = 100ms per word
    [InlineData(1000, 60)]   // 1000 WPM = 60ms per word
    public void IntervalMs_CalculatesCorrectly(int wpm, int expectedMs)
    {
        // Arrange
        var viewModel = new MainWindowViewModel();

        // Act
        viewModel.WPM = wpm;

        // Assert
        Assert.Equal(expectedMs, viewModel.IntervalMs);
    }

    [Fact]
    public void IncreaseSpeed_IncreasesBy50()
    {
        // Arrange
        var viewModel = new MainWindowViewModel();
        var initialSpeed = viewModel.WPM;

        // Act
        viewModel.IncreaseSpeed();

        // Assert
        Assert.Equal(initialSpeed + 50, viewModel.WPM);
    }

    [Fact]
    public void DecreaseSpeed_DecreasesBy50()
    {
        // Arrange
        var viewModel = new MainWindowViewModel();
        viewModel.WPM = 400;

        // Act
        viewModel.DecreaseSpeed();

        // Assert
        Assert.Equal(350, viewModel.WPM);
    }

    [Fact]
    public void DecreaseSpeed_DoesNotGoBelowMinimum()
    {
        // Arrange
        var viewModel = new MainWindowViewModel();
        viewModel.WPM = 50;

        // Act
        viewModel.DecreaseSpeed();

        // Assert
        Assert.Equal(50, viewModel.WPM);
    }

    [Fact]
    public void InputText_UpdatesAndResetsReader()
    {
        // Arrange
        var viewModel = new MainWindowViewModel();
        var newText = "This is a test";

        // Act
        viewModel.InputText = newText;

        // Assert
        Assert.Equal(newText, viewModel.InputText);
        Assert.Equal("This", viewModel.CurrentWord); // Should show first word
        Assert.Equal(0, viewModel.Progress);
    }

    [Fact]
    public void TogglePlayPause_TogglesPlayingState()
    {
        // Arrange
        var viewModel = new MainWindowViewModel();
        var initialState = viewModel.IsPlaying;

        // Act
        viewModel.TogglePlayPause();

        // Assert
        Assert.NotEqual(initialState, viewModel.IsPlaying);

        // Act again
        viewModel.TogglePlayPause();

        // Assert - back to original
        Assert.Equal(initialState, viewModel.IsPlaying);
    }

    [Fact]
    public void Play_StartsReading()
    {
        // Arrange
        var viewModel = new MainWindowViewModel();
        viewModel.InputText = "Test reading text";

        // Act
        viewModel.Play();

        // Assert
        Assert.True(viewModel.IsPlaying);
        Assert.Equal("▶ Play", viewModel.PlayPauseText) == false;
    }

    [Fact]
    public void Pause_StopsReading()
    {
        // Arrange
        var viewModel = new MainWindowViewModel();
        viewModel.Play();

        // Act
        viewModel.Pause();

        // Assert
        Assert.False(viewModel.IsPlaying);
        Assert.Equal("▶ Play", viewModel.PlayPauseText);
    }

    [Fact]
    public void Reset_ResetsToBeginning()
    {
        // Arrange
        var viewModel = new MainWindowViewModel();
        viewModel.InputText = "One Two Three Four";
        viewModel.Play();
        System.Threading.Thread.Sleep(500); // Let it advance

        // Act
        viewModel.Reset();

        // Assert
        Assert.False(viewModel.IsPlaying);
        Assert.Equal(0, viewModel.Progress);
    }

    [Fact]
    public void NextWord_AdvancesWord()
    {
        // Arrange
        var viewModel = new MainWindowViewModel();
        viewModel.InputText = "First Second Third";
        var firstWord = viewModel.CurrentWord;

        // Act
        viewModel.NextWord();

        // Assert
        Assert.NotEqual(firstWord, viewModel.CurrentWord);
    }

    [Fact]
    public void PreviousWord_GoesBackWord()
    {
        // Arrange
        var viewModel = new MainWindowViewModel();
        viewModel.InputText = "First Second Third";
        viewModel.NextWord();
        var secondWord = viewModel.CurrentWord;

        // Act
        viewModel.PreviousWord();

        // Assert
        Assert.NotEqual(secondWord, viewModel.CurrentWord);
    }

    [Fact]
    public void ProgressText_ShowsCorrectFormat()
    {
        // Arrange
        var viewModel = new MainWindowViewModel();
        viewModel.InputText = "One Two Three";

        // Assert
        Assert.Matches(@"\d+ / \d+", viewModel.ProgressText);
        Assert.Contains("/", viewModel.ProgressText);
    }

    [Fact]
    public void ThemeColors_CanBeChanged()
    {
        // Arrange
        var viewModel = new MainWindowViewModel();

        // Act
        viewModel.BackgroundColor = "#FFFFFF";
        viewModel.TextColor = "#000000";
        viewModel.ORPColor = "#FF0000";

        // Assert
        Assert.Equal("#FFFFFF", viewModel.BackgroundColor);
        Assert.Equal("#000000", viewModel.TextColor);
        Assert.Equal("#FF0000", viewModel.ORPColor);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData(null)]
    public void InputText_HandlesEmptyOrNullText(string? text)
    {
        // Arrange
        var viewModel = new MainWindowViewModel();

        // Act
        viewModel.InputText = text ?? "";

        // Assert
        Assert.NotNull(viewModel.InputText);
        Assert.Equal("", viewModel.CurrentWord);
    }

    [Fact]
    public void CloseDefinition_HidesDefinition()
    {
        // Arrange
        var viewModel = new MainWindowViewModel();

        // Act
        viewModel.CloseDefinition();

        // Assert
        Assert.False(viewModel.ShowDefinition);
    }

    [Fact]
    public void ToggleLibrary_TogglesLibraryView()
    {
        // Arrange
        var viewModel = new MainWindowViewModel();
        var initialState = viewModel.ShowLibrary;

        // Act
        viewModel.ToggleLibrary();

        // Assert
        Assert.NotEqual(initialState, viewModel.ShowLibrary);
    }

    [Fact]
    public void SavedBooks_IsInitialized()
    {
        // Arrange & Act
        var viewModel = new MainWindowViewModel();

        // Assert
        Assert.NotNull(viewModel.SavedBooks);
    }

    [Theory]
    [InlineData("Hello world", "Hello")]
    [InlineData("  Multiple   spaces  ", "Multiple")]
    [InlineData("Line1\nLine2\nLine3", "Line1")]
    public void InputText_ParsesFirstWordCorrectly(string input, string expectedFirstWord)
    {
        // Arrange
        var viewModel = new MainWindowViewModel();

        // Act
        viewModel.InputText = input;

        // Assert
        Assert.Equal(expectedFirstWord, viewModel.CurrentWord);
    }
}
