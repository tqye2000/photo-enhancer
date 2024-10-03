import streamlit as st
from PIL import Image
import numpy as np
from enhancer.enhancer import Enhancer
import base64

def get_binary_file_downloader_html(bin_file : bytes, file_label='File'):
    '''
    Generates a link allowing the data in a given bin_file to be downloaded
    in:  bin_file (bytes)
    out: href string
    '''
    b64 = base64.b64encode(bin_file).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_label}">Download {file_label}</a>'
    
    return href

def main():
    #st.title("Image Enhancer App")
    st.header('Image Enhancer App')
    intro_text = """
        - This is a Streamlit app for image enhancement using GANs.
        - This is a Streamlit app for image enhancement using GANs.
        - Upload an image and choose the method to enhance the image.
        - The enhanced image will be displayed and available for download.
        - The available methods are: gfpgan, RestoreFormer, codeformer.
        - You can also choose to enhance the background and upscale the image.\n
        Enjoy!
    """
    st.write(intro_text)

    st.divider()

    image_path = st.file_uploader("Choose file: ", type=['.png', '.jpg', '.jpeg'])

    # app settings
    st.sidebar.header("App Settings:")
    method = st.sidebar.selectbox("Method:", ["gfpgan", "RestoreFormer", "codeformer"])
    background_enhancement = st.sidebar.selectbox("Background enhancement:", ["True", "False"])
    background_enhancement = True if background_enhancement == "True" else False
    upscale = st.sidebar.selectbox("Upscale enhancement:", [2, 4])
    picture_width = st.sidebar.slider('Picture Width', value=320, min_value=100, max_value=500)

    if image_path is not None:
        # Create enhancer
        enhancer = Enhancer(method=method, background_enhancement=background_enhancement, upscale=upscale)
        image = np.array(Image.open(image_path))
        with st.spinner('Wait ...'):
            restored_image = enhancer.enhance(image)
        
        # enhanced image
        final_image = Image.fromarray(restored_image)
        
        # save enhanced image
        file_name = f"{image_path.name.split('.')[0]}_enhanced.jpg"
        #final_image.save(file_name)

        # display code: 2 column view
        col1, col2 = st.columns(2)
        with col1:
            st.header("Input Image")
            st.image(image_path, width=picture_width)
        with col2:
            st.header("Enhanced Image")
            st.image(final_image, width=picture_width)

        # create a download button
        with st.spinner('Wait ...'):
            # Ensure final_image is in bytes format
            if isinstance(final_image, Image.Image):
                import io
                byte_arr = io.BytesIO()
                final_image.save(byte_arr, format='PNG')
                final_image_bytes = byte_arr.getvalue()
            else:
                final_image_bytes = final_image

        st.markdown(get_binary_file_downloader_html(final_image_bytes, file_name), unsafe_allow_html=True)

##############################
if __name__ == "__main__":

    main()

