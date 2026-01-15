import os
import base64
import uuid
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# =========================
# APP INIT
# =========================
app = FastAPI(title="Wayang AI Detection")

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("model/wayang_model.h5")

# =========================
# CLASS NAMES (HARUS SAMA SAAT TRAINING)
# =========================
CLASS_NAMES = [
    "abimanyu","anoman","arjuna","bagong","baladewa","bima","buta",
    "cakil","durna","dursasana","duryudana","gareng","gatotkaca",
    "karna","kresna","nakula_sadewa","patih_sabrang","petruk",
    "puntadewa","semar","sengkuni","togog"
]

# =========================
# DESKRIPSI WAYANG
# =========================
WAYANG_DESCRIPTIONS = {
    "abimanyu": "Abimanyu adalah kesatria muda putra Arjuna dan Subadra. Ia terkenal karena keberaniannya menembus formasi Cakrawyuha dalam perang Bharatayudha, namun gugur secara ksatria di medan perang.",
    "anoman": "Anoman atau Hanoman adalah kera putih sakti, putra Batara Bayu. Ia setia membantu Rama melawan Rahwana dan melambangkan kesetiaan, keberanian, serta pengabdian.",
    "arjuna": "Arjuna adalah kesatria Pandawa yang tampan dan ahli memanah. Ia dikenal tenang, cerdas, dan menjadi murid kesayangan Resi Durna serta sahabat dekat Kresna.",
    "bagong": "Bagong adalah salah satu punakawan, anak Semar, berwatak polos, jujur, dan sering berbicara blak-blakan. Ia melambangkan suara rakyat kecil yang lugas.",
    "baladewa": "Baladewa atau Balarama adalah kakak Kresna, bertubuh besar dan bersenjata gada. Ia dikenal tegas, jujur, dan menjadi simbol kekuatan serta keteguhan hati.",
    "bima": "Bima atau Werkudara adalah kesatria Pandawa bertubuh besar dan sangat kuat. Ia berwatak jujur, tegas, dan setia, serta menjadi andalan dalam pertempuran.",
    "buta": "Buta adalah sebutan bagi bangsa raksasa dalam pewayangan. Mereka melambangkan sifat angkara murka, kekuatan kasar, dan hawa nafsu yang tidak terkendali.",
    "cakil": "Cakil adalah raksasa kecil bertaring dengan watak licik dan agresif. Ia sering menjadi lawan ksatria dalam adegan perang, melambangkan kejahatan dan kesombongan.",
    "durna": "Resi Durna adalah guru para Pandawa dan Kurawa dalam ilmu perang. Meski sakti dan bijak, ia sering terjebak konflik batin dan keberpihakan yang salah.",
    "dursasana": "Dursasana adalah adik Duryudana yang berwatak kasar dan kejam. Ia terkenal karena perannya menghina Drupadi dan melambangkan kebrutalan Kurawa.",
    "duryudana": "Duryudana adalah pemimpin Kurawa, ambisius dan penuh iri hati terhadap Pandawa. Ia melambangkan nafsu kekuasaan dan keserakahan yang membawa kehancuran.",
    "gareng": "Gareng adalah punakawan dengan sifat bijaksana dan rendah hati. Ucapannya sering mengandung petuah moral meski disampaikan dengan cara sederhana.",
    "gatotkaca": "Gatotkaca adalah putra Bima yang memiliki kesaktian dapat terbang dan berbadan baja. Ia gugur sebagai pahlawan dalam Bharatayudha demi membela Pandawa.",
    "karna": "Karna adalah kesatria besar berhati mulia, putra Kunti yang dibesarkan kusir. Meski berpihak pada Kurawa, ia dikenal setia, dermawan, dan ksatria sejati.",
    "kresna": "Kresna adalah raja Dwaraka, titisan Batara Wisnu. Ia menjadi penasihat Pandawa dan lambang kebijaksanaan, kecerdikan, serta pelindung kebenaran.",
    "nakula_sadewa": "Nakula dan Sadewa adalah saudara kembar Pandawa. Nakula terkenal tampan dan ahli pedang, Sadewa bijak dan menguasai ilmu perbintangan.",
    "patih_sabrang": "Patih Sabrang adalah patih dari kerajaan seberang yang biasanya digambarkan sebagai tokoh keras dan agresif, melambangkan ancaman dari luar negeri.",
    "petruk": "Petruk adalah punakawan bertubuh tinggi dengan hidung panjang. Ia cerdas, humoris, dan sering menyampaikan kritik sosial dengan sindiran halus.",
    "puntadewa": "Puntadewa atau Yudistira adalah sulung Pandawa dan raja Amarta. Ia terkenal sangat jujur, adil, dan menjadi simbol dharma atau kebenaran.",
    "semar": "Semar adalah pemimpin punakawan dan jelmaan dewa. Ia berwujud sederhana namun sangat sakti, melambangkan kebijaksanaan, keikhlasan, dan pengayom rakyat.",
    "sengkuni": "Sengkuni adalah paman Kurawa yang licik dan pandai menghasut. Ia menjadi otak berbagai intrik untuk menjatuhkan Pandawa.",
    "togog": "Togog adalah punakawan pihak antagonis, berwatak kasar namun lucu. Ia melambangkan kebodohan dan sifat buruk yang sering menjerumuskan tuannya."
}


# =========================
# TEMPLATE & STATIC
# =========================
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================
# IMAGE → BASE64
# =========================
def img_to_base64(img):
    _, buffer = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buffer).decode()

# =========================
# IMAGE PROCESSING PIPELINE
# =========================
def process_image(image: Image.Image):
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_resize = cv2.resize(img_bgr, (224, 224))

    gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    edges = cv2.Canny(gray, 100, 200)

    # Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Prewitt
    kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    prewitt_x = cv2.filter2D(gray, -1, kernelx)
    prewitt_y = cv2.filter2D(gray, -1, kernely)
    prewitt = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

    # Morphology
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    images = {
        "original": img_to_base64(img_resize),
        "grayscale": img_to_base64(gray),
        "binary": img_to_base64(binary),
        "canny": img_to_base64(edges),
        "sobel": img_to_base64(sobel),
        "prewitt": img_to_base64(prewitt),
        "opening": img_to_base64(opening),
        "closing": img_to_base64(closing)
    }

    return images, img_resize

# =========================
# PREDICTION
# =========================
def predict_image(img):
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[0][idx])

# =========================
# ROUTES
# =========================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")

    images, img_ready = process_image(image)
    class_name, confidence = predict_image(img_ready)

    return JSONResponse({
        "object": "Wayang",
        "class_name": class_name,
        "confidence": round(confidence, 3),
        "description": WAYANG_DESCRIPTIONS.get(class_name, "-"),
        "images": images
    })

# =========================
# AUTO RUN SERVER
# =========================
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Server berjalan di:")
    print("👉 http://127.0.0.1:8000\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
