using SpeedReaderApp.Models;
using SpeedReaderApp.Services;
using System;
using System.IO;
using System.Threading.Tasks;
using Xunit;

namespace SpeedReaderApp.Tests.Services;

public class DatabaseServiceTests : IDisposable
{
    private readonly string _testDbPath;
    private readonly DatabaseService _service;

    public DatabaseServiceTests()
    {
        _testDbPath = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}.db3");
        _service = new DatabaseService(_testDbPath);
    }

    public void Dispose()
    {
        if (File.Exists(_testDbPath))
        {
            File.Delete(_testDbPath);
        }
    }

    [Fact]
    public async Task SaveBookAsync_NewBook_InsertsBook()
    {
        // Arrange
        var book = new SavedBook
        {
            Title = "Test Book",
            Content = "Test content here",
            Source = "Manual",
            TotalWords = 3
        };

        // Act
        var result = await _service.SaveBookAsync(book);

        // Assert
        Assert.True(result > 0);
        Assert.True(book.Id > 0);

        var books = await _service.GetBooksAsync();
        Assert.Single(books);
        Assert.Equal("Test Book", books[0].Title);
    }

    [Fact]
    public async Task SaveBookAsync_ExistingBook_UpdatesBook()
    {
        // Arrange
        var book = new SavedBook
        {
            Title = "Original Title",
            Content = "Content",
            Source = "Manual",
            TotalWords = 1
        };
        await _service.SaveBookAsync(book);

        // Act
        book.Title = "Updated Title";
        var result = await _service.SaveBookAsync(book);

        // Assert
        Assert.True(result > 0);

        var updated = await _service.GetBookAsync(book.Id);
        Assert.NotNull(updated);
        Assert.Equal("Updated Title", updated.Title);
    }

    [Fact]
    public async Task GetBooksAsync_ReturnsAllBooks()
    {
        // Arrange
        await _service.SaveBookAsync(new SavedBook { Title = "Book 1", Content = "Content 1", Source = "PDF", TotalWords = 2 });
        await _service.SaveBookAsync(new SavedBook { Title = "Book 2", Content = "Content 2", Source = "Manual", TotalWords = 2 });
        await _service.SaveBookAsync(new SavedBook { Title = "Book 3", Content = "Content 3", Source = "PDF", TotalWords = 2 });

        // Act
        var books = await _service.GetBooksAsync();

        // Assert
        Assert.Equal(3, books.Count);
    }

    [Fact]
    public async Task GetBookAsync_ValidId_ReturnsBook()
    {
        // Arrange
        var book = new SavedBook
        {
            Title = "Findable Book",
            Content = "Content",
            Source = "Manual",
            TotalWords = 1
        };
        await _service.SaveBookAsync(book);

        // Act
        var found = await _service.GetBookAsync(book.Id);

        // Assert
        Assert.NotNull(found);
        Assert.Equal("Findable Book", found.Title);
    }

    [Fact]
    public async Task GetBookAsync_InvalidId_ReturnsNull()
    {
        // Act
        var found = await _service.GetBookAsync(99999);

        // Assert
        Assert.Null(found);
    }

    [Fact]
    public async Task DeleteBookAsync_RemovesBook()
    {
        // Arrange
        var book = new SavedBook
        {
            Title = "To Delete",
            Content = "Content",
            Source = "Manual",
            TotalWords = 1
        };
        await _service.SaveBookAsync(book);
        var bookId = book.Id;

        // Act
        await _service.DeleteBookAsync(book);

        // Assert
        var books = await _service.GetBooksAsync();
        Assert.Empty(books);

        var deleted = await _service.GetBookAsync(bookId);
        Assert.Null(deleted);
    }

    [Fact]
    public async Task UpdateReadingProgressAsync_UpdatesProgress()
    {
        // Arrange
        var book = new SavedBook
        {
            Title = "Progress Book",
            Content = "One two three four five",
            Source = "Manual",
            TotalWords = 5,
            LastPosition = 0,
            ReadingSpeed = 300
        };
        await _service.SaveBookAsync(book);

        // Act
        await _service.UpdateReadingProgressAsync(book.Id, 3, 500);

        // Assert
        var updated = await _service.GetBookAsync(book.Id);
        Assert.NotNull(updated);
        Assert.Equal(3, updated.LastPosition);
        Assert.Equal(500, updated.ReadingSpeed);
    }

    [Fact]
    public async Task SaveBookAsync_SetsDateAdded()
    {
        // Arrange
        var book = new SavedBook
        {
            Title = "Date Test",
            Content = "Content",
            Source = "Manual",
            TotalWords = 1
        };

        // Act
        await _service.SaveBookAsync(book);

        // Assert
        var saved = await _service.GetBookAsync(book.Id);
        Assert.NotNull(saved);
        Assert.True(saved.DateAdded > DateTime.MinValue);
        Assert.True(saved.LastRead > DateTime.MinValue);
    }

    [Fact]
    public async Task GetBooksAsync_OrdersByLastRead_Descending()
    {
        // Arrange
        var book1 = new SavedBook { Title = "Old Book", Content = "C", Source = "Manual", TotalWords = 1 };
        await _service.SaveBookAsync(book1);
        await Task.Delay(100);

        var book2 = new SavedBook { Title = "Recent Book", Content = "C", Source = "Manual", TotalWords = 1 };
        await _service.SaveBookAsync(book2);

        // Act
        var books = await _service.GetBooksAsync();

        // Assert
        Assert.Equal(2, books.Count);
        Assert.Equal("Recent Book", books[0].Title); // Most recent first
        Assert.Equal("Old Book", books[1].Title);
    }

    [Fact]
    public void GetDatabasePath_ReturnsValidPath()
    {
        // Act
        var path = DatabaseService.GetDatabasePath();

        // Assert
        Assert.NotNull(path);
        Assert.NotEmpty(path);
        Assert.Contains("speedreader.db3", path);
    }
}
