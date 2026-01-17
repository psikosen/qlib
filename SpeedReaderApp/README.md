# Speed Reader - Science-Backed RSVP Reader

A modern, cross-platform speed reading application built with Avalonia UI that implements scientifically-backed speed reading techniques. Read faster while maintaining comprehension using RSVP (Rapid Serial Visual Presentation) technology.

![Platform](https://img.shields.io/badge/platform-Android%20%7C%20iOS%20%7C%20Desktop-blue)
![.NET](https://img.shields.io/badge/.NET-8.0-purple)
![Avalonia](https://img.shields.io/badge/Avalonia-11.1.3-red)

## Features

### 📚 Science-Backed Reading Technology

#### RSVP (Rapid Serial Visual Presentation)
- Displays one word at a time at a fixed focal point
- Eliminates eye movement (saccades) during reading
- Reduces reading fatigue and increases focus
- Scientifically proven effective at 250-500 WPM

#### ORP (Optimal Recognition Point)
- Highlights the optimal letter in each word for eye focus
- Calculated based on word length (typically slightly left of center)
- Improves word recognition speed and accuracy
- Based on research from cognitive psychology and reading science

### ⚡ Speed Control

- **Adjustable Speed**: 50-2000 WPM (Words Per Minute)
- **Speed Presets**: Quick access to 250, 300, 400, 500, and 1000 WPM
- **Fine Control**: +/- 50 WPM increments
- **Science Note**: Research shows best comprehension at 250-350 WPM

### 🎨 Customizable Themes

Four built-in color themes optimized for reading:

1. **Light Theme**: White background, black text
2. **Dark Theme**: Dark background, white text (reduces eye strain)
3. **Sepia Theme**: Warm sepia tones (classic book feel)
4. **High Contrast**: Maximum contrast for accessibility

All themes support custom text and background colors.

### 📖 Built-in Dictionary

- **Double-tap** any word (when paused) to get instant definitions
- Powered by the free [Dictionary API](https://dictionaryapi.dev/)
- Shows multiple definitions, parts of speech, and phonetic pronunciation
- Works offline with cached definitions
- No API key required

### 🎮 Advanced Controls

- **Play/Pause**: Start and stop reading with a tap
- **Word Navigation**: Jump to previous/next word
- **Progress Tracking**: Visual progress bar and word counter
- **Reset**: Start over from the beginning
- **Text Input**: Paste any text to start reading

### 📱 Mobile-Optimized

- **Touch Gestures**:
  - Single tap on reading area: Play/Pause
  - Double tap (when paused): Look up word definition
- **Responsive Design**: Adapts to any screen size
- **Scrollable Controls**: All controls accessible on small screens
- **Portrait/Landscape**: Works in any orientation

## The Science Behind Speed Reading

### What Research Shows

Based on extensive research on RSVP and speed reading:

1. **Optimal Speed Range**: Studies show that comprehension remains high up to **250-350 WPM**
2. **Comprehension Trade-off**: Beyond 500 WPM, comprehension drops significantly
3. **ORP Effectiveness**: Highlighting the optimal recognition point improves reading efficiency by 10-15%
4. **Eye Movement**: Eliminating saccades (eye movements) can save 80% of reading time
5. **Regression Limitation**: The inability to re-read (regression) can hurt comprehension for complex texts

### Why This App Works

- **Fixed Focal Point**: Your eyes stay in one place, reducing fatigue
- **Controlled Pacing**: Consistent word presentation prevents distractions
- **ORP Highlighting**: Guides your eye to the optimal position for recognition
- **Pause Feature**: Allows time for comprehension of difficult passages
- **Progress Tracking**: Helps you know where you are in the text

### Research Sources

- [Rapid serial visual presentation in reading: The case of Spritz](https://www.sciencedirect.com/science/article/abs/pii/S0747563214007663)
- [The Science of Speed Reading](https://www.medicaldaily.com/science-speed-reading-benefits-and-consequences-reading-1000-pages-10-hours-316828)
- [Spritz and other speed reading apps: prose and cons](https://theconversation.com/spritz-and-other-speed-reading-apps-prose-and-cons-24467)
- [Free Dictionary API](https://dictionaryapi.dev/)

## Getting Started

### Prerequisites

- .NET 8.0 SDK or later
- For Android: Android SDK 21+
- For iOS: Xcode 14+ (macOS only)
- For Desktop: Windows 10+, macOS 10.15+, or Linux

### Building the App

#### Desktop (Windows, macOS, Linux)
```bash
cd SpeedReaderApp
dotnet restore
dotnet build
dotnet run
```

#### Android
```bash
cd SpeedReaderApp/SpeedReaderApp.Android
dotnet restore
dotnet build
# Deploy to emulator or device
```

#### iOS
```bash
cd SpeedReaderApp/SpeedReaderApp.iOS
dotnet restore
dotnet build
# Deploy to simulator or device
```

## Usage Guide

### Basic Reading

1. **Load Text**: Paste or type your text in the input box at the bottom
2. **Set Speed**: Choose a preset (recommended: 300 WPM to start) or use +/- buttons
3. **Press Play**: Click the Play button or tap the reading area
4. **Read**: Focus on the center of the screen and let the words flow

### Advanced Features

#### Finding Definitions
1. Pause reading (tap Pause or tap reading area)
2. Double-tap the current word OR click "📖 Define"
3. Read the definition
4. Tap "Close" to resume

#### Customizing Colors
1. Choose a preset theme (Light, Dark, Sepia, High Contrast)
2. Experiment to find what's most comfortable for your eyes
3. Dark themes recommended for extended reading sessions

#### Navigating Text
- Use **⏮ Prev** and **Next ⏭** to move word by word
- Use **🔄 Reset** to start from the beginning
- Progress bar shows your position in the text

### Tips for Best Results

1. **Start Slow**: Begin at 250-300 WPM and gradually increase
2. **Take Breaks**: Pause every 10-15 minutes to reduce eye strain
3. **Choose Right Speed**: Comprehension > Speed. If you're not understanding, slow down
4. **Good Lighting**: Use appropriate themes and ensure good ambient lighting
5. **Practice**: Speed reading is a skill that improves with practice
6. **Complex Text**: Use slower speeds (200-300 WPM) for technical or dense material
7. **Light Reading**: Can use faster speeds (400-600 WPM) for casual reading

## Technical Architecture

### Project Structure

```
SpeedReaderApp/
├── SpeedReaderApp.csproj           # Main shared project
├── App.axaml                       # Application styles
├── App.axaml.cs                    # App initialization
├── Program.cs                      # Entry point
├── ViewModels/
│   └── MainWindowViewModel.cs     # Business logic & state
├── Views/
│   ├── MainWindow.axaml           # Desktop UI
│   ├── MainWindow.axaml.cs        # Desktop code-behind
│   ├── MainView.axaml             # Mobile UI
│   └── MainView.axaml.cs          # Mobile code-behind
├── SpeedReaderApp.Android/        # Android platform
│   ├── MainActivity.cs
│   ├── AndroidManifest.xml
│   └── SpeedReaderApp.Android.csproj
└── SpeedReaderApp.iOS/            # iOS platform
    ├── AppDelegate.cs
    ├── Info.plist
    └── SpeedReaderApp.iOS.csproj
```

### Technologies Used

- **[Avalonia UI](https://avaloniaui.net/)**: Cross-platform XAML-based UI framework
- **.NET 8.0**: Modern, high-performance runtime
- **MVVM Pattern**: Clean separation of concerns
- **ReactiveUI**: Reactive programming for UI
- **Free Dictionary API**: Word definitions without API keys

### Key Components

1. **MainWindowViewModel**: Core business logic
   - Word management and parsing
   - Timer control for RSVP
   - Speed calculations
   - Dictionary API integration
   - Theme management

2. **RSVP Engine**: Precise word timing
   - Calculates interval from WPM
   - Timer-based word advancement
   - Progress tracking

3. **ORP Calculator**: Smart letter highlighting
   - Word length-based calculation
   - Optimal position for eye focus

## Future Enhancements

Potential features for future versions:

- [ ] Reading statistics (words read, time spent, average speed)
- [ ] Reading history and bookmarks
- [ ] Import from files (PDF, EPUB, TXT)
- [ ] Adjustable ORP highlighting intensity
- [ ] Bionic reading mode (highlight first few letters)
- [ ] Focus mode (gradual dimming of completed words)
- [ ] Reading goals and achievements
- [ ] Cloud sync for reading position
- [ ] Text-to-speech integration
- [ ] Multi-language support

## Contributing

Contributions are welcome! Areas that could use improvement:

- Additional color themes
- More dictionary sources
- Reading comprehension tests
- Performance optimizations
- Accessibility features
- Localization

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Avalonia team for the excellent cross-platform framework
- Dictionary API for free word definitions
- Research community for speed reading science
- Spritz and similar apps for pioneering RSVP technology

## Contact & Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Happy Speed Reading!** 📚⚡

Remember: The goal isn't just to read faster—it's to read smarter. Use speed as a tool, not a goal.
