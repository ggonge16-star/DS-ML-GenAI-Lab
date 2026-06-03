import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor Detection & Visualization")
st.markdown("**MobileNetV2 + Grad-CAM + OpenCV | 95.19% Accuracy**")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        r"E:\ml dl project\Open cv\brain_tumor_BEST_95.keras"
    )
    return model

model = load_model()

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

tumor_info = {
    'glioma': {
        'description': 'Glioma is a tumor that occurs in the brain and spinal cord.',
        'severity': '🔴 High Risk',
        'treatment': 'Surgery, Radiation, Chemotherapy'
    },
    'meningioma': {
        'description': 'Meningioma is a tumor that arises from the meninges.',
        'severity': '🟡 Medium Risk',
        'treatment': 'Surgery, Radiation Therapy'
    },
    'notumor': {
        'description': 'No tumor detected in the MRI scan.',
        'severity': '🟢 No Risk',
        'treatment': 'No treatment required'
    },
    'pituitary': {
        'description': 'Pituitary tumor occurs in the pituitary gland.',
        'severity': '🟡 Medium Risk',
        'treatment': 'Surgery, Medication, Radiation'
    }
}

def get_gradcam_heatmap(model, image):
    img_array = tf.cast(np.expand_dims(image, axis=0), tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        preds = model(img_array, training=False)
        pred_index = np.argmax(preds[0].numpy())
        loss = preds[:, pred_index]
    grads = tape.gradient(loss, img_array)
    grads = grads[0]
    heatmap = tf.reduce_mean(tf.abs(grads), axis=-1)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_index, preds[0].numpy()

def is_valid_mri(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:,:,1].mean()
    if saturation > 60:
        return False
    return True

uploaded_file = st.file_uploader(
    "🧠 Brain MRI Image Upload Kara",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (224, 224))

    if not is_valid_mri(resized):
        st.error("⚠️ Valid Brain MRI upload kara!")
        st.stop()

    with st.spinner('🔍 MRI Analyzing...'):
        heatmap, pred_index, probs = get_gradcam_heatmap(model, resized)

    predicted_class = class_names[pred_index]
    confidence = probs[pred_index]

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔍 Prediction Result")
        st.markdown(f"**Tumor Type:** `{predicted_class.upper()}`")
        st.markdown(f"**Confidence:** `{confidence*100:.2f}%`")
        st.markdown(f"**Severity:** {tumor_info[predicted_class]['severity']}")
        st.markdown(f"**Description:** {tumor_info[predicted_class]['description']}")
        st.markdown(f"**Treatment:** {tumor_info[predicted_class]['treatment']}")

    with col2:
        st.markdown("### 📊 All Class Probabilities")
        for name, prob in zip(class_names, probs):
            st.progress(float(prob), text=f"{name}: {prob*100:.2f}%")

    st.markdown("---")
    st.markdown("### 🔬 Visualization")

heatmap_resized = cv2.resize(heatmap, (224, 224))
heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
heatmap_thresh = np.where(heatmap_norm > 0.7, heatmap_norm, 0)
heatmap_smooth = cv2.GaussianBlur(heatmap_thresh, (11, 11), 0)
heatmap_colored = cv2.applyColorMap(
    np.uint8(255 * heatmap_smooth), cv2.COLORMAP_JET
)
heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
overlay = cv2.addWeighted(resized, 0.5, heatmap_colored_rgb, 0.5, 0)

# Contour
contour_img = resized.copy()
heatmap_uint8 = np.uint8(255 * heatmap_resized)
_, thresh = cv2.threshold(heatmap_uint8, 120, 255, cv2.THRESH_BINARY)
kernel = np.ones((11, 11), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
for cnt in contours_sorted[:2]:
    if cv2.contourArea(cnt) > 100:
        cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 3)

col1, col2, col3 = st.columns(3)
with col1:
    st.image(resized, caption="Original MRI", use_container_width=True)
with col2:
    st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=True)
with col3:
    st.image(contour_img, caption="Tumor Location + Contour", use_container_width=True)
    print('owner by Ganesh Gonge❤️')