using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Content;

namespace SpeedReaderApp.Services;

public class PdfService
{
    public async Task<(string content, string title)> ExtractTextFromPdfAsync(Stream pdfStream)
    {
        return await Task.Run(() =>
        {
            try
            {
                using var document = PdfDocument.Open(pdfStream);
                var sb = new StringBuilder();

                // Try to get title from metadata
                var title = "Untitled Document";
                if (document.Information?.Title != null && !string.IsNullOrWhiteSpace(document.Information.Title))
                {
                    title = document.Information.Title;
                }

                // Extract text from all pages
                foreach (var page in document.GetPages())
                {
                    var pageText = page.Text;
                    if (!string.IsNullOrWhiteSpace(pageText))
                    {
                        sb.AppendLine(pageText);
                    }
                }

                var content = sb.ToString();

                // If no title from metadata, try to use first line as title
                if (title == "Untitled Document" && !string.IsNullOrWhiteSpace(content))
                {
                    var firstLine = content.Split('\n').FirstOrDefault(l => !string.IsNullOrWhiteSpace(l));
                    if (firstLine != null && firstLine.Length <= 100)
                    {
                        title = firstLine.Trim();
                    }
                }

                return (content, title);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to read PDF: {ex.Message}", ex);
            }
        });
    }

    public async Task<(string content, string title)> ExtractTextFromPdfAsync(string filePath)
    {
        using var stream = File.OpenRead(filePath);
        return await ExtractTextFromPdfAsync(stream);
    }
}
