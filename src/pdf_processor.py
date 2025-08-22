"""PDF processing and text extraction utilities."""

from pathlib import Path
from typing import TypeAlias
from dataclasses import dataclass, field
import pypdf
import pdfplumber
from tqdm import tqdm

from .config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE

# Type aliases
PageNumber: TypeAlias = int
ChunkId: TypeAlias = str


@dataclass
class TextChunk:
    """Represents a chunk of text from a PDF."""
    
    content: str
    source_file: str
    page_numbers: list[PageNumber]
    chunk_id: ChunkId
    metadata: dict[str, str | int] = field(default_factory=dict)


@dataclass
class PDFProcessor:
    """Handles PDF text extraction and chunking."""
    
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    min_chunk_size: int = MIN_CHUNK_SIZE
    
    def extract_text_pypdf(self, pdf_path: Path) -> dict[PageNumber, str]:
        """Extract text from PDF using pypdf."""
        page_texts: dict[PageNumber, str] = {}
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            for page_num in tqdm(range(total_pages), desc="Extracting pages"):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text.strip():
                    page_texts[page_num + 1] = text
        
        return page_texts
    
    def extract_text_pdfplumber(self, pdf_path: Path) -> dict[PageNumber, str]:
        """Extract text from PDF using pdfplumber (better for tables)."""
        page_texts: dict[PageNumber, str] = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(tqdm(pdf.pages, desc="Extracting pages")):
                text = page.extract_text()
                if text and text.strip():
                    page_texts[page_num + 1] = text
        
        return page_texts
    
    def extract_text(self, pdf_path: Path, method: str = "pypdf") -> dict[PageNumber, str]:
        """Extract text from PDF using specified method."""
        if method == "pypdf":
            return self.extract_text_pypdf(pdf_path)
        elif method == "pdfplumber":
            return self.extract_text_pdfplumber(pdf_path)
        else:
            raise ValueError(f"Unknown extraction method: {method}")
    
    def create_chunks(self, page_texts: dict[PageNumber, str], source_file: str) -> list[TextChunk]:
        """Split text into overlapping chunks."""
        chunks: list[TextChunk] = []
        
        # Combine all pages into one text with page markers
        full_text = ""
        page_map: list[tuple[int, PageNumber]] = []  # (char_position, page_number)
        
        for page_num, text in sorted(page_texts.items()):
            start_pos = len(full_text)
            full_text += f"\n[Page {page_num}]\n{text}"
            page_map.append((start_pos, page_num))
        
        # Create chunks with overlap
        start = 0
        chunk_index = 0
        
        while start < len(full_text):
            # Find end position
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(full_text):
                # Look for sentence end markers
                for marker in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_marker = full_text.rfind(marker, start + self.min_chunk_size, end)
                    if last_marker != -1:
                        end = last_marker + len(marker)
                        break
            
            chunk_text = full_text[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                # Determine which pages this chunk spans
                chunk_pages = self._get_pages_for_chunk(start, end, page_map)
                
                chunk = TextChunk(
                    content=chunk_text,
                    source_file=source_file,
                    page_numbers=chunk_pages,
                    chunk_id=f"{source_file}_chunk_{chunk_index}",
                    metadata={
                        "chunk_index": chunk_index,
                        "start_char": start,
                        "end_char": end,
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start <= 0:
                start = end
        
        return chunks
    
    def _get_pages_for_chunk(
        self, 
        start: int, 
        end: int, 
        page_map: list[tuple[int, PageNumber]]
    ) -> list[PageNumber]:
        """Determine which pages a chunk spans."""
        pages: set[PageNumber] = set()
        
        for i, (pos, page_num) in enumerate(page_map):
            next_pos = page_map[i + 1][0] if i + 1 < len(page_map) else float('inf')
            
            # Check if chunk overlaps with this page
            if start < next_pos and end > pos:
                pages.add(page_num)
        
        return sorted(list(pages))
    
    def process_pdf(self, pdf_path: Path, method: str = "pypdf") -> list[TextChunk]:
        """Complete pipeline to process a PDF into chunks."""
        print(f"Processing PDF: {pdf_path.name}")
        
        # Extract text
        page_texts = self.extract_text(pdf_path, method)
        print(f"Extracted text from {len(page_texts)} pages")
        
        # Create chunks
        chunks = self.create_chunks(page_texts, pdf_path.name)
        print(f"Created {len(chunks)} chunks")
        
        return chunks
