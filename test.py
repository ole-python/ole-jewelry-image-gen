import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from better_profanity import profanity
from autocorrect import Speller
import os


spell = Speller(lang='en')

def correct_spelling(user_prompt):
    """Corrects spelling using Autocorrect library"""
    return " ".join([spell(word) for word in user_prompt.split()])

#########################################################################

negative_prompts = ["nsfw", "nude", "violence", "killing", "murder", "blood", "death", "explicit", "xxx"
,"assault", "weapon", "injury", "war", "torture", "gore", "suicide", "self-harm",'worst quality','normal quality','low quality'
,'low res','blurry','distortion','text','watermark','logo','banner','extra digits','cropped','jpeg artifacts','signature','username'
,'error','sketch','duplicate','ugly','monochrome','horror','geometry','mutation','disgusting','bad anatomy','bad proportions'
,'bad quality','deformed','disconnected limbs','out of frame','out of focus','dehydrated','disfigured','extra arms','extra limbs'
,'extra hands','fused fingers','gross proportions','long neck','jpeg','malformed limbs','mutated','mutated hands','mutated limbs'
,'missing arms','missing fingers','picture frame','poorly drawn hands','poorly drawn face','collage','pixel','pixelated','grainy'
,'color aberration','amputee','autograph','bad illustration','beyond the borders','blank background','body out of frame','boring background'
,'branding','cut off','dismembered','disproportioned','distorted','draft','duplicated features','extra fingers','extra legs','fault','flaw'
,'grains','hazy','identifying mark','improper scale','incorrect physiology','incorrect ratio','indistinct','kitsch','low resolution','macabre'
,'malformed','mark','misshapen','missing hands','missing legs','mistake','morbid','mutilated','off-screen','outside the picture','poorly drawn feet'
,'printed words','render','repellent','replicate','reproduce','revolting dimensions','script','shortened','sign','split image','squint','storyboard'
,'tiling','trimmed','unfocused','unattractive','unnatural pose','unreal engine','unsightly','written language','bad anatomy','bad hands','three hands',
'three legs','bad arms','missing legs','missing arms','poorly drawn face','poorly rendered hands','bad face','fused face','cloned face','worst face',
'three crus','extra crus','fused crus','worst feet','three feet','fused feet','fused thigh','three thigh','extra thigh','worst thigh','missing fingers',
'extra fingers','ugly fingers','long fingers','bad composition','horn','extra eyes','huge eyes','2girl','amputation','disconnected limbs','cartoon','cg',
'3d','unreal','animate','cgi','render','artwork','illustration','3d render','cinema 4d','artstation','octane render','mutated body parts','painting',
'oil painting','2d','sketch','bad photography','bad photo','deviant art','aberrations','abstract','anime','black and white','collapsed','conjoined',
'creative','drawing','extra windows','harsh lighting','jpeg artifacts','low saturation','monochrome','multiple levels','overexposed','oversaturated',
'photoshop','rotten','surreal','twisted','UI','underexposed','unnatural','unreal engine','unrealistic','video game','deformed body features','extra digits',
'extra arms','extra hands','fused fingers','malformed limbs','mutated hands','poorly drawn hands','extra fingers','missing hands','bad hands','three hands',
'fused hands','too many fingers','missing fingers','deformed hands','nsfw','nude','nudity','uncensored','explicit content','cleavage','nipple','adult','porn','xxx',
'nudity','suggestive content','pornographic','sexual content','provocative','explicit scenes','lingerie','seductive','erotic','inappropriate poses','intimate contact','scantily clad','adult themes','violence','gore','blood','brutality','sexy']

profanity.load_censor_words(negative_prompts)


def remove_negative_prompts(user_prompt):
    """Check if prompt contains banned words. If yes, raise an exception."""
    if profanity.contains_profanity(user_prompt):
        return "negative prompt"
    else :
      return "positive prompt"
    


########################################################################

# ✅ Set the local model path
# MODEL_PATH = r"D:\ole-ai-image-gen\model\stable-diffusion-v1-4"
MODEL_PATH = "CompVis/stable-diffusion-v1-4"

# ✅ Check if CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Use float16 on GPU, else use float32 for CPU
dtype = torch.float16 if device == "cuda" else torch.float32

# ✅ Cache model loading to prevent reloading every time
# @st.cache_resource
# def load_model():
#     """Load the Stable Diffusion model from local storage and move to GPU if available."""
#     pipe = StableDiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype=dtype)
#     pipe.to(device)
#     return pipe

@st.cache_resource
def load_model():
    """Load the Stable Diffusion model from Hugging Face and move to CPU (since Streamlit Cloud has no GPU)."""
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe.to("cpu")  # ✅ Streamlit Cloud does NOT support CUDA, so force CPU usage
    return pipe

# ✅ Load the model
st.write("🔄 Loading AI Model... Please wait.")
pipe = load_model()
st.success("✅ Model is ready! Enter a prompt to generate images.")

# 🔹 Streamlit UI
st.title("💍 AI Jewelry Image Generator")
user_prompt = st.text_input("Enter Jewelry Description: (Kindly avoid spelling mistakes for better results)", "")
user_prompt = correct_spelling(user_prompt)
check_negative_prompt = remove_negative_prompts(user_prompt)
if check_negative_prompt == "negative prompt":
    st.error("🚫 Negative or inappropriate content detected in prompt. Please enter a different prompt.")
    st.stop()

final_prompt = f"single jewelery image with customer descriptions as follows: {user_prompt}"

# ✅ Generate Images One by One
if st.button("Generate Images"):
    with st.spinner("Generating Images... Please wait."):
        for i in range(4):  # Generate 4 images
            st.write(f"Generating image {i+1}/4...")

            # ✅ Generate an image
            image = pipe(final_prompt).images[0]

            # ✅ Display the image immediately
            st.image(image, caption=f"Generated Image {i+1}", use_column_width=True)    
