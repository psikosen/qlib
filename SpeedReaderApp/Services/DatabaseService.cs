using SQLite;
using SpeedReaderApp.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace SpeedReaderApp.Services;

public class DatabaseService
{
    private readonly SQLiteAsyncConnection _database;

    public DatabaseService(string dbPath)
    {
        _database = new SQLiteAsyncConnection(dbPath);
        _database.CreateTableAsync<SavedBook>().Wait();
    }

    public Task<List<SavedBook>> GetBooksAsync()
    {
        return _database.Table<SavedBook>()
            .OrderByDescending(b => b.LastRead)
            .ToListAsync();
    }

    public Task<SavedBook?> GetBookAsync(int id)
    {
        return _database.Table<SavedBook>()
            .Where(b => b.Id == id)
            .FirstOrDefaultAsync();
    }

    public Task<int> SaveBookAsync(SavedBook book)
    {
        if (book.Id != 0)
        {
            return _database.UpdateAsync(book);
        }
        else
        {
            book.DateAdded = DateTime.Now;
            book.LastRead = DateTime.Now;
            return _database.InsertAsync(book);
        }
    }

    public Task<int> DeleteBookAsync(SavedBook book)
    {
        return _database.DeleteAsync(book);
    }

    public Task<int> UpdateReadingProgressAsync(int bookId, int position, int wpm)
    {
        return _database.ExecuteAsync(
            "UPDATE saved_books SET LastPosition = ?, LastRead = ?, ReadingSpeed = ? WHERE Id = ?",
            position, DateTime.Now, wpm, bookId);
    }

    public static string GetDatabasePath()
    {
        var folder = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        return Path.Combine(folder, "speedreader.db3");
    }
}
