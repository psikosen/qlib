using SpeedReaderApp.Services;
using System;
using System.IO;
using System.Threading.Tasks;
using Xunit;

namespace SpeedReaderApp.Tests.Services;

public class PdfServiceTests
{
    private readonly PdfService _service;

    public PdfServiceTests()
    {
        _service = new PdfService();
    }

    [Fact]
    public async Task ExtractTextFromPdfAsync_InvalidStream_ThrowsException()
    {
        // Arrange
        using var stream = new MemoryStream();
        stream.Write(new byte[] { 0x00, 0x01, 0x02 }); // Invalid PDF data
        stream.Position = 0;

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => _service.ExtractTextFromPdfAsync(stream));
    }

    [Fact]
    public async Task ExtractTextFromPdfAsync_EmptyStream_ThrowsException()
    {
        // Arrange
        using var stream = new MemoryStream();

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => _service.ExtractTextFromPdfAsync(stream));
    }

    [Fact]
    public async Task ExtractTextFromPdfAsync_NonExistentFile_ThrowsException()
    {
        // Arrange
        var nonExistentPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.pdf");

        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(
            () => _service.ExtractTextFromPdfAsync(nonExistentPath));
    }

    // Note: Testing with actual PDF files would require sample PDFs
    // In a real project, you would include test PDFs in the test resources
}
