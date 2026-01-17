# Speed Reader App - Testing Guide

## ⚠️ Critical Testing Requirements

This document provides comprehensive testing instructions to ensure the app works correctly. We must be adversarial in our testing - assume things will break and test accordingly.

## Build and Run Tests

### Prerequisites
- .NET 8.0 SDK or later
- Visual Studio 2022 / VS Code / Rider

### Build the Application

```bash
# Restore packages
cd SpeedReaderApp
dotnet restore

# Build the main app
dotnet build

# Build tests
cd ../SpeedReaderApp.Tests
dotnet restore
dotnet build
```

### Run Unit Tests

```bash
cd SpeedReaderApp.Tests
dotnet test --verbosity normal

# Run with coverage
dotnet test /p:CollectCoverage=true /p:CoverageOutputFormat=lcov
```

### Run the Desktop App

```bash
cd SpeedReaderApp
dotnet run
```

## Manual Testing Checklist

### ✅ Core Reading Functionality

#### Test 1: Basic Text Reading
- [ ] Paste text into input box
- [ ] Click Play
- [ ] Verify words appear one at a time
- [ ] Verify words change at correct interval (300 WPM = ~200ms per word)
- [ ] Click Pause
- [ ] Verify reading stops
- [ ] Click Play again
- [ ] Verify reading resumes from same position

**Expected Behavior**: Smooth word-by-word display with accurate timing

**Potential Issues to Check**:
- Timer not starting
- Words not updating
- Timing incorrect
- Memory leaks from timer

#### Test 2: Speed Controls
- [ ] Set speed to 250 WPM
- [ ] Start reading and verify timing (~240ms per word)
- [ ] While reading, increase speed to 500 WPM
- [ ] Verify timing changes immediately (~120ms per word)
- [ ] Try decrease speed
- [ ] Try all presets (250, 300, 400, 500, 1000)
- [ ] Try setting speed above 2000 - should clamp to 2000
- [ ] Try setting speed below 50 - should clamp to 50

**Expected Behavior**: Speed changes take effect immediately; bounds enforced

**Potential Issues**:
- Timer not updating when speed changes
- Invalid speed values
- Division by zero if WPM is 0

#### Test 3: Navigation Controls
- [ ] Load text "One Two Three Four Five"
- [ ] Click Next Word multiple times
- [ ] Verify word advances correctly
- [ ] Click Previous Word
- [ ] Verify word goes back
- [ ] Try Previous Word at beginning - should stay at first word
- [ ] Try Next Word at end - should stay at last word
- [ ] Click Reset
- [ ] Verify returns to first word

**Expected Behavior**: Accurate word navigation with boundary checks

### ✅ PDF Loading

#### Test 4: Valid PDF Upload
- [ ] Click "📄 Load PDF"
- [ ] Select a valid PDF file with text
- [ ] Verify loading indicator appears
- [ ] Verify "Reading PDF..." message shows
- [ ] Wait for completion
- [ ] Verify text appears in reader
- [ ] Verify word count is correct
- [ ] Verify PDF is saved to database
- [ ] Check library - PDF should appear

**Expected Behavior**: PDF loads successfully, text extracted, saved to database

**Potential Issues**:
- PDF with no text (scanned images only)
- Corrupted PDF file
- Very large PDF (100+ pages)
- PDF with special characters/encodings
- Memory issues with large files

#### Test 5: Invalid PDF Handling
- [ ] Try to load a non-PDF file renamed to .pdf
- [ ] Verify error message appears
- [ ] Verify app doesn't crash
- [ ] Try to load an empty file
- [ ] Try to load a corrupted PDF

**Expected Behavior**: Graceful error handling, no crashes

#### Test 6: Image-Only PDF
- [ ] Load a PDF that contains only scanned images (no selectable text)
- [ ] Verify appropriate error message
- [ ] Verify loading indicator disappears

**Expected Behavior**: Clear error message about no text found

### ✅ Database Functionality

#### Test 7: Save and Load Books
- [ ] Paste text and start reading
- [ ] Save as a book (need to add UI for this)
- [ ] Load another text
- [ ] Open library
- [ ] Select the saved book
- [ ] Verify original text loads
- [ ] Verify reading position is restored
- [ ] Verify reading speed is restored

**Expected Behavior**: Books persist across sessions, resume from last position

**Potential Issues**:
- Database file permissions
- Concurrent access issues
- Corrupted database
- Missing database file

#### Test 8: Multiple Books
- [ ] Save 5 different books
- [ ] Verify all appear in library
- [ ] Verify sorted by last read (most recent first)
- [ ] Open each book
- [ ] Verify correct content loads

**Expected Behavior**: Multiple books managed correctly

#### Test 9: Delete Books
- [ ] Open library
- [ ] Delete a book
- [ ] Verify book removed from library
- [ ] Verify database updated
- [ ] If currently reading deleted book, verify graceful handling

**Expected Behavior**: Books deleted cleanly

### ✅ Dictionary Lookup

#### Test 10: Definition Lookup - Online
- [ ] Pause on a word
- [ ] Click "📖 Define" or double-tap
- [ ] Verify "Loading definition..." appears
- [ ] Verify definition loads from API
- [ ] Verify definition shows word, pronunciation, and meanings
- [ ] Try multiple different words
- [ ] Click Close
- [ ] Verify popup disappears

**Expected Behavior**: Definitions load successfully

**Potential Issues**:
- No internet connection
- API rate limiting
- API down
- Unusual words not in dictionary
- Special characters in words

#### Test 11: Definition Lookup - Offline
- [ ] Disconnect from internet
- [ ] Pause on a word
- [ ] Click "📖 Define"
- [ ] Verify appropriate error message
- [ ] Verify suggests checking internet connection

**Expected Behavior**: Clear offline error message

#### Test 12: Invalid Words
- [ ] Test with numbers: "123"
- [ ] Test with symbols: "@#$"
- [ ] Test with very long strings
- [ ] Test with empty word

**Expected Behavior**: Appropriate error messages

### ✅ Theme and Color Customization

#### Test 13: Theme Switching
- [ ] Try Light theme
- [ ] Verify background is white, text is black
- [ ] Start reading - verify readable
- [ ] Try Dark theme
- [ ] Verify background is dark, text is white
- [ ] Try Sepia theme
- [ ] Try High Contrast theme
- [ ] Verify themes persist during reading

**Expected Behavior**: All themes work correctly and are readable

### ✅ Progress Tracking

#### Test 14: Progress Bar
- [ ] Load text with 10 words
- [ ] Start reading
- [ ] Watch progress bar
- [ ] Verify it advances correctly
- [ ] Verify it reaches 100% at end
- [ ] Verify word counter shows "X / 10"

**Expected Behavior**: Accurate progress tracking

### ✅ Edge Cases and Stress Tests

#### Test 15: Empty Text
- [ ] Clear all text from input
- [ ] Click Play
- [ ] Verify app doesn't crash
- [ ] Verify appropriate handling

**Expected Behavior**: Graceful handling of empty input

#### Test 16: Very Long Text
- [ ] Paste a 10,000+ word document
- [ ] Verify it loads
- [ ] Start reading
- [ ] Verify performance is acceptable
- [ ] Check memory usage
- [ ] Navigate to end

**Expected Behavior**: Handles large texts without performance issues

#### Test 17: Special Characters
- [ ] Test with emojis: "Hello 😀 World 🎉"
- [ ] Test with unicode: "Héllo Wørld Привет"
- [ ] Test with symbols: "Test (with) [brackets] {and} <symbols>"

**Expected Behavior**: All characters display correctly

#### Test 18: Rapid Speed Changes
- [ ] Start reading at 300 WPM
- [ ] Rapidly click + and - buttons
- [ ] Try changing presets rapidly
- [ ] Verify no crashes
- [ ] Verify timer updates correctly

**Expected Behavior**: Stable performance under rapid changes

#### Test 19: Rapid Play/Pause
- [ ] Start reading
- [ ] Rapidly click Play/Pause 20 times
- [ ] Verify state is consistent
- [ ] Verify no timer leaks
- [ ] Check resource usage

**Expected Behavior**: Stable state management

### ✅ Platform-Specific Testing

#### Test 20: Desktop Window Management
- [ ] Resize window - verify layout adapts
- [ ] Minimize and restore
- [ ] Close and reopen app
- [ ] Verify database persists

**Expected Behavior**: Proper window management

#### Test 21: Mobile Touch Gestures (Android/iOS)
- [ ] Single tap reading area - verify play/pause
- [ ] Double tap word - verify definition lookup
- [ ] Scroll controls - verify accessible
- [ ] Portrait/landscape - verify layout adapts

**Expected Behavior**: Touch gestures work correctly

## Bug Report Template

When you find a bug, document it:

```markdown
### Bug: [Short Description]

**Severity**: Critical / High / Medium / Low

**Steps to Reproduce**:
1.
2.
3.

**Expected Behavior**:


**Actual Behavior**:


**Environment**:
- OS:
- .NET Version:
- App Version:

**Screenshots/Logs**:


**Potential Cause**:


**Suggested Fix**:

```

## Known Issues to Test For

Based on code review, these are potential issues:

### 1. **Timer Disposal**
- **Issue**: Timer might not be properly disposed on rapid play/pause
- **Test**: Rapidly toggle play/pause 50 times, check memory
- **Location**: `MainWindowViewModel.cs:208-215`

### 2. **ORP Calculation Edge Cases**
- **Issue**: ORP calculation might fail on single-character words or empty strings
- **Test**: Try words of length 0, 1, 2, 100
- **Location**: `MainWindowViewModel.cs:412-421`

### 3. **Progress Calculation**
- **Issue**: Division by zero if _words.Length is 0
- **Test**: Clear text and start reading
- **Location**: `MainWindowViewModel.cs:376`

### 4. **Database Concurrency**
- **Issue**: Multiple operations might conflict
- **Test**: Save multiple books rapidly
- **Location**: `DatabaseService.cs`

### 5. **PDF Memory Usage**
- **Issue**: Large PDFs might cause OOM
- **Test**: Load a 500+ page PDF
- **Location**: `PdfService.cs:14-45`

### 6. **HTTP Timeout**
- **Issue**: Dictionary API might timeout
- **Test**: Look up words with slow connection
- **Location**: `MainWindowViewModel.cs:258-333`

## Performance Benchmarks

Target performance metrics:

| Metric | Target | Test Method |
|--------|--------|-------------|
| Startup Time | < 2 seconds | Measure from launch to UI ready |
| PDF Load (10 pages) | < 3 seconds | Time from selection to text ready |
| PDF Load (100 pages) | < 15 seconds | Time from selection to text ready |
| Dictionary Lookup | < 2 seconds | Time to show definition |
| Memory (idle) | < 100 MB | Task manager |
| Memory (10k words) | < 150 MB | Task manager |
| Memory (100k words) | < 300 MB | Task manager |
| CPU (reading at 300 WPM) | < 5% | Task manager |

## Accessibility Testing

- [ ] Test with screen reader
- [ ] Test keyboard navigation
- [ ] Test with high contrast theme
- [ ] Test font scaling
- [ ] Test color blind friendly themes

## Security Considerations

- [ ] Verify no sensitive data in logs
- [ ] Verify database is local only
- [ ] Verify HTTPS for dictionary API
- [ ] Verify no credentials stored
- [ ] Check file permissions on database

## Conclusion

This app must be tested thoroughly before release. The core functionality is solid, but edge cases and error handling must be verified. Be adversarial - try to break it!

**Remember**: Users will do unexpected things. Test for them!
