#!/usr/bin/env python3
import os
import glob
import base64
import time
import shutil
import zipfile
import re
import logging
import asyncio
import random
from io import BytesIO
from datetime import datetime
import pytz
from dataclasses import dataclass
from typing import Optional

import streamlit as st
import pandas as pd
import torch
import fitz
import requests
import aiofiles
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from openai import OpenAI  # Updated import for new API

# --- OpenAI Setup ---
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    organization=os.getenv('OPENAI_ORG_ID')
)

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
log_records = []
class LogCaptureHandler(logging.Handler):
    def emit(self, record):
        log_records.append(record)
logger.addHandler(LogCaptureHandler())

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="AI Vision & SFT Titans 🚀",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://huggingface.co/awacke1',
        'Report a Bug': 'https://huggingface.co/spaces/awacke1',
        'About': "AI Vision & SFT Titans: PDFs, OCR, Image Gen, Line Drawings, Custom Diffusion, and SFT on CPU! 🌌"
    }
)

# --- Session State Defaults ---
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'builder' not in st.session_state:
    st.session_state['builder'] = None
if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False
if 'processing' not in st.session_state:
    st.session_state['processing'] = {}
if 'asset_checkboxes' not in st.session_state:
    st.session_state['asset_checkboxes'] = {}
if 'downloaded_pdfs' not in st.session_state:
    st.session_state['downloaded_pdfs'] = {}
if 'unique_counter' not in st.session_state:
    st.session_state['unique_counter'] = 0
if 'selected_model_type' not in st.session_state:
    st.session_state['selected_model_type'] = "Causal LM"
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = "None"
if 'cam0_file' not in st.session_state:
    st.session_state['cam0_file'] = None
if 'cam1_file' not in st.session_state:
    st.session_state['cam1_file'] = None

# --- Model & Diffusion DataClasses ---
@dataclass
class ModelConfig:
    name: str
    base_model: str
    size: str
    domain: Optional[str] = None
    model_type: str = "causal_lm"
    @property
    def model_path(self):
        return f"models/{self.name}"

@dataclass
class DiffusionConfig:
    name: str
    base_model: str
    size: str
    domain: Optional[str] = None
    @property
    def model_path(self):
        return f"diffusion_models/{self.name}"

# --- Model Builders ---
class ModelBuilder:
    def __init__(self):
        self.config = None
        self.model = None
        self.tokenizer = None
        self.jokes = ["Why did the AI go to therapy? Too many layers to unpack! 😂",
                      "Training complete! Time for a binary coffee break. ☕"]
    def load_model(self, model_path: str, config: Optional[ModelConfig] = None):
        with st.spinner(f"Loading {model_path}... ⏳"):
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if config:
                self.config = config
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        st.success(f"Model loaded! 🎉 {random.choice(self.jokes)}")
        return self
    def save_model(self, path: str):
        with st.spinner("Saving model... 💾"):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        st.success(f"Model saved at {path}! ✅")

class DiffusionBuilder:
    def __init__(self):
        self.config = None
        self.pipeline = None
    def load_model(self, model_path: str, config: Optional[DiffusionConfig] = None):
        with st.spinner(f"Loading diffusion model {model_path}... ⏳"):
            self.pipeline = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32).to("cpu")
            if config:
                self.config = config
        st.success("Diffusion model loaded! 🎨")
        return self
    def save_model(self, path: str):
        with st.spinner("Saving diffusion model... 💾"):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.pipeline.save_pretrained(path)
        st.success(f"Diffusion model saved at {path}! ✅")
    def generate(self, prompt: str):
        return self.pipeline(prompt, num_inference_steps=20).images[0]

# --- Utility Functions ---
def generate_filename(sequence, ext="png"):
    timestamp = time.strftime("%d%m%Y%H%M%S")
    return f"{sequence}_{timestamp}.{ext}"

def pdf_url_to_filename(url):
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', url)
    return f"{safe_name}.pdf"

def get_download_link(file_path, mime_type="application/pdf", label="Download"):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:{mime_type};base64,{b64}" download="{os.path.basename(file_path)}">{label}</a>'

def zip_directory(directory_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory_path):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.dirname(directory_path)))

def get_model_files(model_type="causal_lm"):
    path = "models/*" if model_type == "causal_lm" else "diffusion_models/*"
    dirs = [d for d in glob.glob(path) if os.path.isdir(d)]
    return dirs if dirs else ["None"]

def get_gallery_files(file_types=["png", "pdf"]):
    return sorted(list(set([f for ext in file_types for f in glob.glob(f"*.{ext}")])))  # Deduplicate files

def get_pdf_files():
    return sorted(glob.glob("*.pdf"))

def download_pdf(url, output_path):
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
    return False

# --- Original PDF Snapshot & OCR Functions ---
async def process_pdf_snapshot(pdf_path, mode="single"):
    start_time = time.time()
    status = st.empty()
    status.text(f"Processing PDF Snapshot ({mode})... (0s)")
    try:
        doc = fitz.open(pdf_path)
        output_files = []
        if mode == "single":
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            output_file = generate_filename("single", "png")
            pix.save(output_file)
            output_files.append(output_file)
        elif mode == "twopage":
            for i in range(min(2, len(doc))):
                page = doc[i]
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                output_file = generate_filename(f"twopage_{i}", "png")
                pix.save(output_file)
                output_files.append(output_file)
        elif mode == "allpages":
            for i in range(len(doc)):
                page = doc[i]
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                output_file = generate_filename(f"page_{i}", "png")
                pix.save(output_file)
                output_files.append(output_file)
        doc.close()
        elapsed = int(time.time() - start_time)
        status.text(f"PDF Snapshot ({mode}) completed in {elapsed}s!")
        update_gallery()
        return output_files
    except Exception as e:
        status.error(f"Failed to process PDF: {str(e)}")
        return []

async def process_ocr(image, output_file):
    start_time = time.time()
    status = st.empty()
    status.text("Processing GOT-OCR2_0... (0s)")
    tokenizer = AutoTokenizer.from_pretrained("ucaslcl/GOT-OCR2_0", trust_remote_code=True)
    model = AutoModel.from_pretrained("ucaslcl/GOT-OCR2_0", trust_remote_code=True, torch_dtype=torch.float32).to("cpu").eval()
    temp_file = f"temp_{int(time.time())}.png"
    image.save(temp_file)
    result = model.chat(tokenizer, temp_file, ocr_type='ocr')
    os.remove(temp_file)
    elapsed = int(time.time() - start_time)
    status.text(f"GOT-OCR2_0 completed in {elapsed}s!")
    async with aiofiles.open(output_file, "w") as f:
        await f.write(result)
    update_gallery()
    return result

async def process_image_gen(prompt, output_file):
    start_time = time.time()
    status = st.empty()
    status.text("Processing Image Gen... (0s)")
    if st.session_state['builder'] and isinstance(st.session_state['builder'], DiffusionBuilder) and st.session_state['builder'].pipeline:
        pipeline = st.session_state['builder'].pipeline
    else:
        pipeline = StableDiffusionPipeline.from_pretrained("OFA-Sys/small-stable-diffusion-v0", torch_dtype=torch.float32).to("cpu")
    gen_image = pipeline(prompt, num_inference_steps=20).images[0]
    elapsed = int(time.time() - start_time)
    status.text(f"Image Gen completed in {elapsed}s!")
    gen_image.save(output_file)
    update_gallery()
    return gen_image

# --- Updated Function: Process an image (PIL) with a custom prompt using GPT ---
def process_image_with_prompt(image, prompt, model="gpt-4o-mini", detail="auto"):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_str}",
                    "detail": detail  # Added detail parameter
                }
            }
        ]
    }]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing image with GPT: {str(e)}"

# --- Updated Function: Process text with GPT ---
def process_text_with_prompt(text, prompt, model="gpt-4o-mini"):
    messages = [{"role": "user", "content": prompt + "\n\n" + text}]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing text with GPT: {str(e)}"

# --- Sidebar Setup ---
st.sidebar.subheader("Gallery Settings")
if 'gallery_size' not in st.session_state:
    st.session_state['gallery_size'] = 2  # Default value
st.session_state['gallery_size'] = st.sidebar.slider(
    "Gallery Size", 
    1, 10, st.session_state['gallery_size'], 
    key="gallery_size_slider"
)

# --- Updated Gallery Function ---
def update_gallery():
    all_files = get_gallery_files()
    if all_files:
        st.sidebar.subheader("Asset Gallery 📸📖")
        cols = st.sidebar.columns(2)
        for idx, file in enumerate(all_files[:st.session_state['gallery_size']]):
            with cols[idx % 2]:
                st.session_state['unique_counter'] += 1
                unique_id = st.session_state['unique_counter']
                if file.endswith('.png'):
                    st.image(Image.open(file), caption=os.path.basename(file), use_container_width=True)
                else:
                    doc = fitz.open(file)
                    pix = doc[0].get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    st.image(img, caption=os.path.basename(file), use_container_width=True)
                    doc.close()
                checkbox_key = f"asset_{file}_{unique_id}"
                st.session_state['asset_checkboxes'][file] = st.checkbox(
                    "Use for SFT/Input", 
                    value=st.session_state['asset_checkboxes'].get(file, False), 
                    key=checkbox_key
                )
                mime_type = "image/png" if file.endswith('.png') else "application/pdf"
                st.markdown(get_download_link(file, mime_type, "Snag It! 📥"), unsafe_allow_html=True)
                if st.button("Zap It! 🗑️", key=f"delete_{file}_{unique_id}"):
                    os.remove(file)
                    st.session_state['asset_checkboxes'].pop(file, None)
                    st.sidebar.success(f"Asset {os.path.basename(file)} vaporized! 💨")
                    st.rerun()

update_gallery()

# --- Sidebar Logs & History ---
st.sidebar.subheader("Action Logs 📜")
with st.sidebar:
    for record in log_records:
        st.write(f"{record.asctime} - {record.levelname} - {record.message}")
st.sidebar.subheader("History 📜")
with st.sidebar:
    for entry in st.session_state['history']:
        st.write(entry)

# --- Create Tabs ---
tabs = st.tabs([
    "Camera Snap 📷",
    "Download PDFs 📥",
    "Test OCR 🔍",
    "Build Titan 🌱",
    "Test Image Gen 🎨",
    "PDF Process 📄",
    "Image Process 🖼️",
    "MD Gallery 📚"
])
(tab_camera, tab_download, tab_ocr, tab_build, tab_imggen, tab_pdf_process, tab_image_process, tab_md_gallery) = tabs

# === Tab: Camera Snap ===
with tab_camera:
    st.header("Camera Snap 📷")
    st.subheader("Single Capture")
    cols = st.columns(2)
    with cols[0]:
        cam0_img = st.camera_input("Take a picture - Cam 0", key="cam0")
        if cam0_img:
            filename = generate_filename("cam0")
            if st.session_state['cam0_file'] and os.path.exists(st.session_state['cam0_file']):
                os.remove(st.session_state['cam0_file'])
            with open(filename, "wb") as f:
                f.write(cam0_img.getvalue())
            st.session_state['cam0_file'] = filename
            entry = f"Snapshot from Cam 0: {filename}"
            if entry not in st.session_state['history']:
                st.session_state['history'] = [e for e in st.session_state['history'] if not e.startswith("Snapshot from Cam 0:")] + [entry]
            st.image(Image.open(filename), caption="Camera 0", use_container_width=True)
            logger.info(f"Saved snapshot from Camera 0: {filename}")
            update_gallery()
    with cols[1]:
        cam1_img = st.camera_input("Take a picture - Cam 1", key="cam1")
        if cam1_img:
            filename = generate_filename("cam1")
            if st.session_state['cam1_file'] and os.path.exists(st.session_state['cam1_file']):
                os.remove(st.session_state['cam1_file'])
            with open(filename, "wb") as f:
                f.write(cam1_img.getvalue())
            st.session_state['cam1_file'] = filename
            entry = f"Snapshot from Cam 1: {filename}"
            if entry not in st.session_state['history']:
                st.session_state['history'] = [e for e in st.session_state['history'] if not e.startswith("Snapshot from Cam 1:")] + [entry]
            st.image(Image.open(filename), caption="Camera 1", use_container_width=True)
            logger.info(f"Saved snapshot from Camera 1: {filename}")
            update_gallery()

# === Tab: Download PDFs ===
with tab_download:
    st.header("Download PDFs 📥")
    if st.button("Examples 📚"):
        example_urls = [
            "https://arxiv.org/pdf/2308.03892",
            "https://arxiv.org/pdf/1912.01703",
            "https://arxiv.org/pdf/2408.11039",
            "https://arxiv.org/pdf/2109.10282",
            "https://arxiv.org/pdf/2112.10752",
            "https://arxiv.org/pdf/2308.11236",
            "https://arxiv.org/pdf/1706.03762",
            "https://arxiv.org/pdf/2006.11239",
            "https://arxiv.org/pdf/2305.11207",
            "https://arxiv.org/pdf/2106.09685",
            "https://arxiv.org/pdf/2005.11401",
            "https://arxiv.org/pdf/2106.10504"
        ]
        st.session_state['pdf_urls'] = "\n".join(example_urls)
    
    url_input = st.text_area("Enter PDF URLs (one per line)", value=st.session_state.get('pdf_urls', ""), height=200)
    if st.button("Robo-Download 🤖"):
        urls = url_input.strip().split("\n")
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_urls = len(urls)
        existing_pdfs = get_pdf_files()
        for idx, url in enumerate(urls):
            if url:
                output_path = pdf_url_to_filename(url)
                status_text.text(f"Fetching {idx + 1}/{total_urls}: {os.path.basename(output_path)}...")
                if output_path not in existing_pdfs:
                    if download_pdf(url, output_path):
                        st.session_state['downloaded_pdfs'][url] = output_path
                        logger.info(f"Downloaded PDF from {url} to {output_path}")
                        entry = f"Downloaded PDF: {output_path}"
                        if entry not in st.session_state['history']:
                            st.session_state['history'].append(entry)
                        st.session_state['asset_checkboxes'][output_path] = True
                    else:
                        st.error(f"Failed to nab {url} 😿")
                else:
                    st.info(f"Already got {os.path.basename(output_path)}! Skipping... 🐾")
                    st.session_state['downloaded_pdfs'][url] = output_path
                progress_bar.progress((idx + 1) / total_urls)
        status_text.text("Robo-Download complete! 🚀")
        update_gallery()
    mode = st.selectbox("Snapshot Mode", ["Single Page (High-Res)", "Two Pages (High-Res)", "All Pages (High-Res)"], key="download_mode")
    if st.button("Snapshot Selected 📸"):
        selected_pdfs = [path for path in get_gallery_files() if path.endswith('.pdf') and st.session_state['asset_checkboxes'].get(path, False)]
        if selected_pdfs:
            for pdf_path in selected_pdfs:
                mode_key = {"Single Page (High-Res)": "single", "Two Pages (High-Res)": "twopage", "All Pages (High-Res)": "allpages"}[mode]
                snapshots = asyncio.run(process_pdf_snapshot(pdf_path, mode_key))
                for snapshot in snapshots:
                    st.image(Image.open(snapshot), caption=snapshot, use_container_width=True)
                    st.session_state['asset_checkboxes'][snapshot] = True
            update_gallery()
        else:
            st.warning("No PDFs selected for snapshotting! Check some boxes in the sidebar.")

# === Tab: Test OCR ===
with tab_ocr:
    st.header("Test OCR 🔍")
    all_files = get_gallery_files()
    if all_files:
        if st.button("OCR All Assets 🚀"):
            full_text = "# OCR Results\n\n"
            for file in all_files:
                if file.endswith('.png'):
                    image = Image.open(file)
                else:
                    doc = fitz.open(file)
                    pix = doc[0].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    doc.close()
                output_file = generate_filename(f"ocr_{os.path.basename(file)}", "txt")
                result = asyncio.run(process_ocr(image, output_file))
                full_text += f"## {os.path.basename(file)}\n\n{result}\n\n"
                entry = f"OCR Test: {file} -> {output_file}"
                if entry not in st.session_state['history']:
                    st.session_state['history'].append(entry)
            md_output_file = f"full_ocr_{int(time.time())}.md"
            with open(md_output_file, "w") as f:
                f.write(full_text)
            st.success(f"Full OCR saved to {md_output_file}")
            st.markdown(get_download_link(md_output_file, "text/markdown", "Download Full OCR Markdown"), unsafe_allow_html=True)
        selected_file = st.selectbox("Select Image or PDF", all_files, key="ocr_select")
        if selected_file:
            if selected_file.endswith('.png'):
                image = Image.open(selected_file)
            else:
                doc = fitz.open(selected_file)
                pix = doc[0].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                doc.close()
            st.image(image, caption="Input Image", use_container_width=True)
            if st.button("Run OCR 🚀", key="ocr_run"):
                output_file = generate_filename("ocr_output", "txt")
                st.session_state['processing']['ocr'] = True
                result = asyncio.run(process_ocr(image, output_file))
                entry = f"OCR Test: {selected_file} -> {output_file}"
                if entry not in st.session_state['history']:
                    st.session_state['history'].append(entry)
                st.text_area("OCR Result", result, height=200, key="ocr_result")
                st.success(f"OCR output saved to {output_file}")
                st.session_state['processing']['ocr'] = False
            if selected_file.endswith('.pdf') and st.button("OCR All Pages 🚀", key="ocr_all_pages"):
                doc = fitz.open(selected_file)
                full_text = f"# OCR Results for {os.path.basename(selected_file)}\n\n"
                for i in range(len(doc)):
                    pix = doc[i].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    output_file = generate_filename(f"ocr_page_{i}", "txt")
                    result = asyncio.run(process_ocr(image, output_file))
                    full_text += f"## Page {i + 1}\n\n{result}\n\n"
                    entry = f"OCR Test: {selected_file} Page {i + 1} -> {output_file}"
                    if entry not in st.session_state['history']:
                        st.session_state['history'].append(entry)
                md_output_file = f"full_ocr_{os.path.basename(selected_file)}_{int(time.time())}.md"
                with open(md_output_file, "w") as f:
                    f.write(full_text)
                st.success(f"Full OCR saved to {md_output_file}")
                st.markdown(get_download_link(md_output_file, "text/markdown", "Download Full OCR Markdown"), unsafe_allow_html=True)
    else:
        st.warning("No assets in gallery yet. Use Camera Snap or Download PDFs!")

# === Tab: Build Titan ===
with tab_build:
    st.header("Build Titan 🌱")
    model_type = st.selectbox("Model Type", ["Causal LM", "Diffusion"], key="build_type")
    base_model = st.selectbox("Select Tiny Model", 
        ["HuggingFaceTB/SmolLM-135M", "Qwen/Qwen1.5-0.5B-Chat"] if model_type == "Causal LM" else 
        ["OFA-Sys/small-stable-diffusion-v0", "stabilityai/stable-diffusion-2-base"])
    model_name = st.text_input("Model Name", f"tiny-titan-{int(time.time())}")
    domain = st.text_input("Target Domain", "general")
    if st.button("Download Model ⬇️"):
        config = (ModelConfig if model_type == "Causal LM" else DiffusionConfig)(name=model_name, base_model=base_model, size="small", domain=domain)
        builder = ModelBuilder() if model_type == "Causal LM" else DiffusionBuilder()
        builder.load_model(base_model, config)
        builder.save_model(config.model_path)
        st.session_state['builder'] = builder
        st.session_state['model_loaded'] = True
        st.session_state['selected_model_type'] = model_type
        st.session_state['selected_model'] = config.model_path
        entry = f"Built {model_type} model: {model_name}"
        if entry not in st.session_state['history']:
            st.session_state['history'].append(entry)
        st.success(f"Model downloaded and saved to {config.model_path}! 🎉")
        st.rerun()

# === Tab: Test Image Gen ===
with tab_imggen:
    st.header("Test Image Gen 🎨")
    all_files = get_gallery_files()
    if all_files:
        selected_file = st.selectbox("Select Image or PDF", all_files, key="gen_select")
        if selected_file:
            if selected_file.endswith('.png'):
                image = Image.open(selected_file)
            else:
                doc = fitz.open(selected_file)
                pix = doc[0].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                doc.close()
            st.image(image, caption="Reference Image", use_container_width=True)
            prompt = st.text_area("Prompt", "Generate a neon superhero version of this image", key="gen_prompt")
            if st.button("Run Image Gen 🚀", key="gen_run"):
                output_file = generate_filename("gen_output", "png")
                st.session_state['processing']['gen'] = True
                result = asyncio.run(process_image_gen(prompt, output_file))
                entry = f"Image Gen Test: {prompt} -> {output_file}"
                if entry not in st.session_state['history']:
                    st.session_state['history'].append(entry)
                st.image(result, caption="Generated Image", use_container_width=True)
                st.success(f"Image saved to {output_file}")
                st.session_state['processing']['gen'] = False
    else:
        st.warning("No images or PDFs in gallery yet. Use Camera Snap or Download PDFs!")
    update_gallery()

# === Updated Tab: PDF Process ===
with tab_pdf_process:
    st.header("PDF Process")
    st.subheader("Upload PDFs for GPT-based text extraction")
    gpt_models = ["gpt-4o", "gpt-4o-mini"]  # Add more vision-capable models as needed
    selected_gpt_model = st.selectbox("Select GPT Model", gpt_models, key="pdf_gpt_model")
    detail_level = st.selectbox("Detail Level", ["auto", "low", "high"], key="pdf_detail_level")
    uploaded_pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, key="pdf_process_uploader")
    view_mode = st.selectbox("View Mode", ["Single Page", "Double Page"], key="pdf_view_mode")
    if st.button("Process Uploaded PDFs", key="process_pdfs"):
        combined_text = ""
        for pdf_file in uploaded_pdfs:
            pdf_bytes = pdf_file.read()
            temp_pdf_path = f"temp_{pdf_file.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_bytes)
            try:
                doc = fitz.open(temp_pdf_path)
                st.write(f"Processing {pdf_file.name} with {len(doc)} pages")
                if view_mode == "Single Page":
                    for i, page in enumerate(doc):
                        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        st.image(img, caption=f"{pdf_file.name} Page {i+1}")
                        gpt_text = process_image_with_prompt(img, "Extract the electronic text from image", model=selected_gpt_model, detail=detail_level)
                        combined_text += f"\n## {pdf_file.name} - Page {i+1}\n\n{gpt_text}\n"
                else:  # Double Page
                    pages = list(doc)
                    for i in range(0, len(pages), 2):
                        if i+1 < len(pages):
                            pix1 = pages[i].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                            img1 = Image.frombytes("RGB", [pix1.width, pix1.height], pix1.samples)
                            pix2 = pages[i+1].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                            img2 = Image.frombytes("RGB", [pix2.width, pix2.height], pix2.samples)
                            total_width = img1.width + img2.width
                            max_height = max(img1.height, img2.height)
                            combined_img = Image.new("RGB", (total_width, max_height))
                            combined_img.paste(img1, (0, 0))
                            combined_img.paste(img2, (img1.width, 0))
                            st.image(combined_img, caption=f"{pdf_file.name} Pages {i+1}-{i+2}")
                            gpt_text = process_image_with_prompt(combined_img, "Extract the electronic text from image", model=selected_gpt_model, detail=detail_level)
                            combined_text += f"\n## {pdf_file.name} - Pages {i+1}-{i+2}\n\n{gpt_text}\n"
                        else:
                            pix = pages[i].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            st.image(img, caption=f"{pdf_file.name} Page {i+1}")
                            gpt_text = process_image_with_prompt(img, "Extract the electronic text from image", model=selected_gpt_model, detail=detail_level)
                            combined_text += f"\n## {pdf_file.name} - Page {i+1}\n\n{gpt_text}\n"
                doc.close()
            except Exception as e:
                st.error(f"Error processing {pdf_file.name}: {str(e)}")
            finally:
                os.remove(temp_pdf_path)
        output_filename = generate_filename("processed_pdf", "md")
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(combined_text)
        st.success(f"PDF processing complete. MD file saved as {output_filename}")
        st.markdown(get_download_link(output_filename, "text/markdown", "Download Processed PDF MD"), unsafe_allow_html=True)

# === Updated Tab: Image Process ===
with tab_image_process:
    st.header("Image Process")
    st.subheader("Upload Images for GPT-based OCR")
    gpt_models = ["gpt-4o", "gpt-4o-mini"]  # Add more vision-capable models as needed
    selected_gpt_model = st.selectbox("Select GPT Model", gpt_models, key="img_gpt_model")
    detail_level = st.selectbox("Detail Level", ["auto", "low", "high"], key="img_detail_level")
    prompt_img = st.text_input("Enter prompt for image processing", "Extract the electronic text from image", key="img_process_prompt")
    uploaded_images = st.file_uploader("Upload image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="image_process_uploader")
    if st.button("Process Uploaded Images", key="process_images"):
        combined_text = ""
        for img_file in uploaded_images:
            try:
                img = Image.open(img_file)
                st.image(img, caption=img_file.name)
                gpt_text = process_image_with_prompt(img, prompt_img, model=selected_gpt_model, detail=detail_level)
                combined_text += f"\n## {img_file.name}\n\n{gpt_text}\n"
            except Exception as e:
                st.error(f"Error processing image {img_file.name}: {str(e)}")
        output_filename = generate_filename("processed_image", "md")
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(combined_text)
        st.success(f"Image processing complete. MD file saved as {output_filename}")
        st.markdown(get_download_link(output_filename, "text/markdown", "Download Processed Image MD"), unsafe_allow_html=True)

# === Updated Tab: MD Gallery ===
with tab_md_gallery:
    st.header("MD Gallery and GPT Processing")
    gpt_models = ["gpt-4o", "gpt-4o-mini"]  # Add more vision-capable models as needed
    selected_gpt_model = st.selectbox("Select GPT Model", gpt_models, key="md_gpt_model")
    md_files = sorted(glob.glob("*.md"))
    if md_files:
        st.subheader("Individual File Processing")
        cols = st.columns(2)
        for idx, md_file in enumerate(md_files):
            with cols[idx % 2]:
                st.write(md_file)
                if st.button(f"Process {md_file}", key=f"process_md_{md_file}"):
                    try:
                        with open(md_file, "r", encoding="utf-8") as f:
                            content = f.read()
                        prompt_md = "Summarize this into markdown outline with emojis and number the topics 1..12"
                        result_text = process_text_with_prompt(content, prompt_md, model=selected_gpt_model)
                        st.markdown(result_text)
                        output_filename = generate_filename(f"processed_{os.path.splitext(md_file)[0]}", "md")
                        with open(output_filename, "w", encoding="utf-8") as f:
                            f.write(result_text)
                        st.markdown(get_download_link(output_filename, "text/markdown", f"Download {output_filename}"), unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error processing {md_file}: {str(e)}")
        st.subheader("Batch Processing")
        st.write("Select MD files to combine and process:")
        selected_md = {}
        for md_file in md_files:
            selected_md[md_file] = st.checkbox(md_file, key=f"checkbox_md_{md_file}")
        batch_prompt = st.text_input("Enter batch processing prompt", "Summarize this into markdown outline with emojis and number the topics 1..12", key="batch_prompt")
        if st.button("Process Selected MD Files", key="process_batch_md"):
            combined_content = ""
            for md_file, selected in selected_md.items():
                if selected:
                    try:
                        with open(md_file, "r", encoding="utf-8") as f:
                            combined_content += f"\n## {md_file}\n" + f.read() + "\n"
                    except Exception as e:
                        st.error(f"Error reading {md_file}: {str(e)}")
            if combined_content:
                result_text = process_text_with_prompt(combined_content, batch_prompt, model=selected_gpt_model)
                st.markdown(result_text)
                output_filename = generate_filename("batch_processed_md", "md")
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(result_text)
                st.success(f"Batch processing complete. MD file saved as {output_filename}")
                st.markdown(get_download_link(output_filename, "text/markdown", "Download Batch Processed MD"), unsafe_allow_html=True)
            else:
                st.warning("No MD files selected.")
    else:
        st.warning("No MD files found.")
