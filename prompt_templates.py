prompt_extract_text_from_images = """
you are given a image, each image contains text.
please extract the text from the image. (OCR task)

the image's text are mainly about the recruitment, like CV, Cover letter
at this stage, you are required to extract the text from the images.
you are strictly required to extract the ALL ALL all all (total) the text from the images.
here are the following image:
you should notice, your answer should contain only(but all) the text extracted from the images. 
(you are encouraged only support to answer text in the image)

"""

ANALYSIS_CORE = """
you are a professional HR, based on the text, you need to make the text clear and human-readable {structured_cv_text}, {structured_cv_text}.
based on the text extracted, make the file about recruitment, much clear and human-readable.
{extracted_cv_text}
notice, this part is to let you 这一步的关键在于对提取的文本进行整理和优化，确保内容的准确性和条理性。输出格式为 {structured_cv_text}
不需要增加/减少文字。
作为提示一般来说 有 求职的基本信息，求职目标，技术，教育，工作经历，项目经验，其他有关的一些维度等等。
"""

