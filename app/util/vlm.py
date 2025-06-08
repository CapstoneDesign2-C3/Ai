from transformers import AutoModel, AutoTokenizer
import torch
import cv2
import glob
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

class VLM:
  def __init__(self):
    
    #VLM
    self.model_path = 'OpenGVLab/VideoChat-Flash-Qwen2-7B_res448'
    self.tokenzier = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
    #로컬에서 돌리기 위해 잠깐 수정하였음.
    self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).to(torch.bfloat16).cuda()

    self.image_processor = self.model.get_vision_tower().image_processor
    self.max_num_frames = 512
    self.generation_config = dict(
        do_sample=False,
        temperature=0.0,
        max_new_tokens=1024,
        top_p=0,
        num_beams=1
    )
    #Translation
    self.translation_model_name = "facebook/mbart-large-50-many-to-many-mmt"
    self.translation_model = MBartForConditionalGeneration.from_pretrained(self.translation_model_name)
    self.translation_tokenizer = MBart50TokenizerFast.from_pretrained(self.translation_model_name)

  def vlm_summary(self, angle, video_data):
    question = "영상은 현재 " + angle + "을 비추고 있는 카메라 영상입니다." \
    "서정적인 표현을 제외하고 영상 내용을 간결하게 요약해 주세요. " \
    "자동차 사고 및 사상자가 포함된 경우 상황을 포함하여 요약해 주세요. " \
    "답변은 영어로 해주세요.".format(angle)

    output = self.model.chat(video_path=video_data, 
                                            tokenizer=self.tokenizer, 
                                            user_prompt=question, 
                                            max_num_frames=self.max_num_frames, 
                                            generation_config=self.generation_config)

    return self.translate(output)

  def vlm_feature(self, image_data):
    question = "주어진 동영상에서 사물에 대한 인상을 알려주세요. 답변은 영어로 해주세요."
    output = self.model.chat(video_path=self.convert(image_data), 
                                            tokenizer=self.tokenizer, 
                                            user_prompt=question, 
                                            max_num_frames=self.max_num_frames, 
                                            generation_config=self.generation_config)

    return self.translate(output)

  def convert(self, image_data):
    image_files = glob.glob(image_data)
    fps = 24.0
    output_file = 'output_video.mp4'
    img = cv2.imread(image_files[0])
    height, width, _ = img.shape
    video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image_file in image_files:
      img = cv2.imread(image_file)
      video_writer.write(img)
    
    video_writer.release()
    return output_file

  def translate(self, output):
    self.translation_tokenizer.src_lang = "en_XX"
    encoded_hi = self.translation_tokenizer(output, return_tensors="pt")
    generated_tokens = self.translation_model.generate(
        **encoded_hi,
        forced_bos_token_id=self.translation_tokenizer.lang_code_to_id["ko_KR"]
    )
    result = self.translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return result