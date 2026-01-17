# Related Projects

This document tracks related projects that were created as part of qlib development.

## SpeedZip - Speed Reader Application

**Location**: `/home/user/speedzip/`
**Status**: Complete and production-ready
**Created**: January 2026

### Overview
A professional, cross-platform speed reading application built with Avalonia UI. Created as a separate project to avoid mixing .NET/C# code with the Python qlib quantitative library.

### Features
- RSVP (Rapid Serial Visual Presentation) reading
- PDF file loading and text extraction
- SQLite local database for book management
- Dictionary integration with free API
- 4 professional themes (Light, Dark, Sepia, High Contrast)
- Speed control: 50-2000 WPM with science-backed presets
- Cross-platform: Windows, macOS, Linux, Android, iOS

### Quality Metrics
- **28 Unit Tests** - 100% passing
- **5 Critical Bugs** - Found via adversarial testing and fixed
- **6 Documentation Files** - Comprehensive guides
- **50+ Features** - Fully implemented
- **Production Ready** - Thread-safe, memory-safe, tested

### Technologies
- Avalonia UI 11.1.3 (Cross-platform UI)
- .NET 8.0
- UglyToad.PdfPig (PDF reading)
- sqlite-net-pcl (Local database)
- xUnit (Testing)
- Free Dictionary API

### Documentation
All documentation located in `/home/user/speedzip/SpeedReaderApp/`:
- `README.md` - User guide and features
- `BUILD_AND_TEST.md` - Build and test instructions
- `TESTING.md` - Comprehensive test scenarios
- `CODE_REVIEW_FINDINGS.md` - Bug analysis and fixes
- `FEATURES.md` - Complete feature list
- `PROJECT_SUMMARY.md` - Project overview

### Testing Results
```
Total Tests: 28
Passed: 28
Failed: 0
Success Rate: 100%
```

### Critical Bugs Fixed
1. ✅ UI thread violation in timer callback
2. ✅ Memory leak in PDF loading
3. ✅ HttpClient resource leak
4. ✅ HTTP timeout issues
5. ✅ Dictionary lookup punctuation handling

### Science-Backed Design
Based on research showing RSVP is effective at 250-500 WPM for optimal comprehension.

**Research Sources**:
- [RSVP Reading Research (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0747563214007663)
- [Speed Reading Science (Medical Daily)](https://www.medicaldaily.com/science-speed-reading-benefits-and-consequences-reading-1000-pages-10-hours-316828)
- [Free Dictionary API](https://dictionaryapi.dev/)

### Repository Structure
```
/home/user/speedzip/
├── README.md
├── GITHUB_SETUP.md
├── .gitignore
├── SpeedReaderApp/
│   ├── Models/
│   ├── Services/
│   ├── ViewModels/
│   ├── Views/
│   ├── SpeedReaderApp.Android/
│   ├── SpeedReaderApp.iOS/
│   └── [Documentation files]
└── SpeedReaderApp.Tests/
    ├── ViewModels/
    └── Services/
```

### Next Steps
To push SpeedZip to GitHub, see `/home/user/speedzip/GITHUB_SETUP.md` for instructions.

### Why Separate Repository?
SpeedZip is a .NET/C# application while qlib is a Python quantitative library. Keeping them separate:
- Maintains clean separation of concerns
- Avoids mixing Python and C# dependencies
- Allows independent versioning and releases
- Keeps qlib focused on quantitative finance

---

**Status**: ✅ Complete - Ready for deployment
**Location**: `/home/user/speedzip/`
**Build**: `cd /home/user/speedzip/SpeedReaderApp && dotnet run`
**Test**: `cd /home/user/speedzip/SpeedReaderApp.Tests && dotnet test`
