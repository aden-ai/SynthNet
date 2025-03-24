import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

def generate_synthetic_data(generator, num_samples=1000, latent_dim=128):
    noise = tf.random.normal([num_samples, latent_dim])
    generated_images = generator(noise, training=False)
    generated_images = ((generated_images + 1) * 127.5).numpy().astype('uint8')
    return generated_images

def plot_generated_images(images, dataset_type="mnist"):
    num_images = len(images)
    cols = min(5, num_images)
    rows = (num_images - 1) // cols + 1
    
    plt.figure(figsize=(15, 3 * rows))
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        # For RGB images (Fashion MNIST might be displayed in color)
        if images.shape[-1] == 3:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def analyze_generated_images(images):
    return {
        "Total Images": len(images),
        "Mean Pixel Value": np.mean(images),
        "Pixel Value Std Dev": np.std(images),
        "Min Pixel Value": np.min(images),
        "Max Pixel Value": np.max(images)
    }

def load_generator_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

def main():
    st.set_page_config(
        page_title="SynthNet: MNIST-Dataset Generator",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;} /* Hides the main menu */
        footer {visibility: hidden;} /* Hides the footer */
        header {visibility: hidden;} /* Hides the header */
        .css-1d391kg {visibility: hidden;} /* Hides the status indicator */
        .css-1v3fvcr {visibility: hidden;} /* Hides the Streamlit watermark */
        .css-1v0mbdj {visibility: hidden;} /* Hides the overall container */
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
        color: #2C3E50;
    }
    .sidebar .sidebar-content {
        background-color: #F0F2F6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🤖 SynthNet: MNIST-Dataset Generator")
    st.markdown("""
    ### Synthetic Data Generation using Deep Convolutional Generative Adversarial Networks
    
    Generate realistic synthetic images from MNIST datasets using advanced machine learning techniques.
    """)

    image = Image.open("images/banner.png")
    st.image(image, caption="Sample Generated Images")
    
    # Initialize session state for generators
    if 'current_generator' not in st.session_state:
        st.session_state.current_generator = None
    if 'current_dataset' not in st.session_state:
        st.session_state.current_dataset = None
    
    # Dataset descriptions
    dataset_info = {
        "mnist": {
            "title": "MNIST Digits",
            "description": "Classic handwritten digits (0-9)",
            "model_path": "models/generator.keras",
            "icon": "🔢"
        },
        "fmnist": {
            "title": "Fashion MNIST (fine-tuned)",
            "description": "Fashion items like shirts, shoes, etc.",
            "model_path": "models/fmnist_generator_model.keras",
            "icon": "👕"
        },
        "emnist": {
            "title": "Extended MNIST (fine-tuned)",
            "description": "Handwritten digits and letters",
            "model_path": "models/emnist_generator_model.keras",
            "icon": "🔤"
        }
    }
        
    st.markdown("---")
    st.header("Image Generation")

    st.sidebar.header("🛠️ Model Selection")
    
    # Create radio buttons for dataset selection
    dataset_options = [f"{info['icon']} {info['title']}" for ds, info in dataset_info.items()]
    dataset_keys = list(dataset_info.keys())
    
    selected_dataset_index = st.sidebar.radio(
        "Select Dataset",
        options=range(len(dataset_options)),
        format_func=lambda x: dataset_options[x]
    )
    
    selected_dataset = dataset_keys[selected_dataset_index]
    
    # Display dataset description
    st.sidebar.markdown(f"**Description:** {dataset_info[selected_dataset]['description']}")
    
    # Load model button
    if st.session_state.current_dataset != selected_dataset or st.session_state.current_generator is None:
        if st.sidebar.button(f"🚀 Load {dataset_info[selected_dataset]['title']} Generator"):
            try:
                model_path = dataset_info[selected_dataset]['model_path']
                st.session_state.current_generator = load_generator_model(model_path)
                st.session_state.current_dataset = selected_dataset
                st.sidebar.success(f"✅ {dataset_info[selected_dataset]['title']} model loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"🚨 Model Initialization Failed: {e}")
    else:
        st.sidebar.success(f"✅ {dataset_info[selected_dataset]['title']} model loaded")
        
        # Optional unload button
        if st.sidebar.button("🔌 Unload Model"):
            st.session_state.current_generator = None
            st.session_state.current_dataset = None
            st.rerun()

    st.sidebar.markdown("<br>", unsafe_allow_html=True)  # Adds a line break

    num_images = st.sidebar.slider(
        "Number of Images to Generate", 
        min_value=1, 
        max_value=1000, 
        value=10
    )

    # Display dataset-specific title in the main area
    if st.session_state.current_dataset:
        st.subheader(f"Generate {dataset_info[st.session_state.current_dataset]['icon']} {dataset_info[st.session_state.current_dataset]['title']}")
    
    if st.button("Generate Synthetic Images"):
        if st.session_state.current_generator is not None:
            try:
                images = generate_synthetic_data(
                    st.session_state.current_generator, 
                    num_images
                )
                st.subheader("Generated Images")
                image_buffer = plot_generated_images(images, st.session_state.current_dataset)
                st.image(image_buffer)
                
                st.subheader("Image Analysis")
                stats = analyze_generated_images(images)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    for key, value in list(stats.items())[:1]:
                        st.metric(key, f"{int(value)}")
                with col2:
                    for key, value in list(stats.items())[1:2]:
                        st.metric(key, f"{value:.2f}")
                with col3:
                    for key, value in list(stats.items())[2:3]:
                        st.metric(key, f"{value:.2f}")
                with col4:
                    for key, value in list(stats.items())[3:4]:
                        st.metric(key, f"{value:.2f}")
                with col5:
                    for key, value in list(stats.items())[4:]:
                        st.metric(key, f"{value:.2f}")
            
            except Exception as e:
                st.error(f"Error Generating Images: {e}")
        else:
            st.warning(f"Please load the {dataset_info[selected_dataset]['title']} model first!")
    
    st.sidebar.header("📖 About SynthNet")
    st.sidebar.info(f"""
    SynthNet is a Deep Convolutional GAN for generating synthetic images from MNIST datasets.
    
    Available Datasets:
    - {dataset_info['mnist']['icon']} {dataset_info['mnist']['title']}: {dataset_info['mnist']['description']}
    - {dataset_info['fmnist']['icon']} {dataset_info['fmnist']['title']}: {dataset_info['fmnist']['description']}
    - {dataset_info['emnist']['icon']} {dataset_info['emnist']['title']}: {dataset_info['emnist']['description']}
    
    Key Features:
    - High-quality image generation
    - Multiple MNIST dataset support
    - Configurable latent space
    - Advanced machine learning techniques
    """)

main()
