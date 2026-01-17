using SQLite;
using System;

namespace SpeedReaderApp.Models;

[Table("saved_books")]
public class SavedBook
{
    [PrimaryKey, AutoIncrement]
    public int Id { get; set; }

    [MaxLength(500)]
    public string Title { get; set; } = string.Empty;

    public string Content { get; set; } = string.Empty;

    [MaxLength(100)]
    public string Source { get; set; } = string.Empty; // "PDF", "Text", "Manual"

    public DateTime DateAdded { get; set; }

    public DateTime LastRead { get; set; }

    public int LastPosition { get; set; } // Last word index

    public int TotalWords { get; set; }

    public int ReadingSpeed { get; set; } // Last used WPM

    public bool IsCompleted { get; set; }
}
