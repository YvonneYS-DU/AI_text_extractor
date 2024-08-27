import fitz  # PyMuPDF
import base64
import asyncio
import os
import aiofiles
import docx
import pandas as pd
from config import api_key
from llmwrapper import Agents
from prompt_templates import prompt_extract_text_from_images

class FileLoader:
    """Class for extracting text and images from uploaded files (PDF, txt, png, jpg)."""

    @staticmethod
    def __machine_pdf_images(file_path, encode_base64=False):
        """
        Extract images from a PDF file and optionally encode them to Base64.
        
        Args:
        file_path (str): The path to the PDF file.
        encode_base64 (bool): If True, encode images to Base64.
        
        Returns:
        list: List of binary image data or Base64 encoded strings.
        """
        images = []
        with fitz.open(file_path) as doc:
            for page in doc:
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_data = base_image["image"]
                    if encode_base64:
                        image_data = base64.b64encode(image_data).decode('utf-8')
                    images.append(image_data)
        return images
    
    @staticmethod
    def __machine_pdf_text(file_path):
        """
        Extract text from a PDF file.
        
        Args:
        file_path (str): The path to the PDF file.
        
        Returns:
        str: Extracted text.
        """
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    
    @staticmethod
    async def __llm_ocr_text_extractor(images_list, api_key=api_key, text_prompt_template=prompt_extract_text_from_images, detail_parameter='auto', temperature=0.3):
        """
        Extract text from images using an LLM model.
        
        Args:
        images_list (list): List of images in Base64 format.
        api_key (str): API key for the LLM model.
        text_prompt_template (str): Prompt template for extracting text from images.
        detail_parameter (str): Detail level parameter for the LLM model.
        temperature (float): Temperature parameter for the LLM model.
        
        Returns:
        list: List of responses from the LLM model.
        """
        tasks = []  # List to store asynchronous tasks
        
        for image in images_list:
            # Asynchronously create an LLM object
            llm = Agents.llm_model(model='openai', model_name='gpt-4o-mini-2024-07-18', temperature=temperature, api_key=api_key)
            
            # Create a chain object
            chain = Agents.agent_chain_create(
                model=llm,
                text_prompt_template=text_prompt_template,
                image_prompt_template=True
            )
            
            # Asynchronously invoke the chain
            task = asyncio.create_task(Agents.chain_batch_generator_async(chain, {'base64_image': image, 'detail_parameter': detail_parameter}))
            tasks.append(task)
        
        # Wait for all tasks to complete
        responses = await asyncio.gather(*tasks)
        print(responses)
        return responses
    
    @staticmethod
    async def _process_pdf_file(file_path, encode_base64=True):
        """
        Separate image and machine-generated PDFs and extract text from PDFs.
        
        Args:
        file_path (str): The path to the PDF file.
        encode_base64 (bool): If True, encode images to Base64.
        
        Returns:
        list/str: Extracted text or responses from the LLM model.
        """
        with fitz.open(file_path) as doc:
            for page_number in range(min(3, len(doc))):  # Only check the first three pages
                page = doc[page_number]
                text = page.get_text()
                if len(text.strip()) > 3:  # If there's more than 3 characters, it's likely machine-generated
                    return FileLoader.__machine_pdf_text(file_path)  # Machine-generated PDF
                else:
                    image64 = FileLoader.__machine_pdf_images(file_path, encode_base64)  # Image PDF
                    text_responses = await FileLoader.__llm_ocr_text_extractor(image64)
                    return text_responses
    
    @staticmethod
    async def PDF_extractor(file_path):
        """
        Extract text and images from a PDF file.
        
        Args:
        file_path (str): The path to the PDF file.
        
        Returns:
        list/str: Extracted text or responses from the LLM model.
        """
        result = await FileLoader._process_pdf_file(file_path)
        return result

    @staticmethod
    async def TXT_extractor(file_path):
        """
        Extract text from a TXT file.
        
        Args:
        file_path (str): The path to the TXT file.
        
        Returns:
        str: Extracted text.
        """
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        return content

    @staticmethod
    def WORD_extractor(file_path):
        """
        Extract text from a Word document.
        
        Args:
        file_path (str): The path to the Word document.
        
        Returns:
        str: Extracted text.
        """

        doc = docx.Document(file_path)
        content = [paragraph.text for paragraph in doc.paragraphs]
        return "\n".join(content)

    @staticmethod
    def EXCEL_extractor(file_path):
        """
        Extract text from an Excel file.
        
        Args:
        file_path (str): The path to the Excel file.
        
        Returns:
        str: Extracted text in CSV format.
        """

        df = pd.read_excel(file_path)
        return df.to_csv(index=False)

    @staticmethod
    async def IMAGE_extractor(file_path):
        """
        Extract text from an image file.
        
        Args:
        file_path (str): The path to the image file.
        
        Returns:
        list: Responses from the LLM model.
        """
        image_list = []
        with open(file_path, 'rb') as f:
            image_list.append(base64.b64encode(f.read()).decode('utf-8'))
        text_responses = await FileLoader.__llm_ocr_text_extractor(image_list)
        return text_responses
    
    @staticmethod
    async def file_handler(file_path):
        """
        Handle file extraction based on file type.
        
        Args:
        file_path (str): The path to the file.
        
        Returns:
        list/str: Extracted text or responses from the LLM model.
        """
        _, file_extension = os.path.splitext(file_path)
        if file_extension == '.pdf':
            return await FileLoader.PDF_extractor(file_path)
        elif file_extension == '.txt':
            return await FileLoader.TXT_extractor(file_path)
        elif file_extension in ['.doc', '.docx']:
            return FileLoader.WORD_extractor(file_path)
        elif file_extension in ['.xls', '.xlsx']:
            return FileLoader.EXCEL_extractor(file_path)
        elif file_extension in ['.jpg', '.jpeg', '.png', '.gif']:
            return await FileLoader.IMAGE_extractor(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    @staticmethod
    async def process_files(file_paths):
        """
        Process multiple files concurrently.
        
        Args:
        file_paths (list): List of file paths to be processed.
        
        Returns:
        list: List of extracted texts or responses from the LLM model.
        """
        tasks = [FileLoader.file_handler(file_path) for file_path in file_paths]
        results = await asyncio.gather(*tasks)
        return results