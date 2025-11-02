# real_ai.py (CPU-only, cosine similarity scoring fixed)
import open_clip
import torch
from PIL import Image
import cv2
import queue
import threading
import torch.nn.functional as F

device = torch.device("cpu")

# === Load CLIP model ===
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14', pretrained='laion2b_s32b_b82k'
)
model.eval().to(device)
tokenizer = open_clip.get_tokenizer('ViT-L-14')

print(f"CLIP READY ON: {device}")

MAKES = [
    "Toyota", "Honda", "Ford", "Chevrolet", "Nissan", "BMW", "Mercedes-Benz", "Audi",
    "Volkswagen", "Hyundai", "Kia", "Subaru", "Mazda", "Lexus", "Jeep", "Ram",
    "GMC", "Dodge", "Cadillac", "Tesla", "Porsche", "Volvo", "Acura", "Infiniti"
]

TOP3_MODELS = {
    "Toyota": ["Camry", "Corolla", "RAV4"],
    "Honda": ["Civic", "Accord", "CR-V"],
    "Ford": ["F-150", "Mustang", "Explorer"],
    "Chevrolet": ["Silverado", "Malibu", "Equinox"],
    "Nissan": ["Altima", "Rogue", "Sentra"],
    "BMW": ["3 Series", "X5", "5 Series"],
    "Mercedes-Benz": ["C-Class", "GLE", "E-Class"],
    "Audi": ["A4", "Q5", "A6"],
    "Volkswagen": ["Golf", "Passat", "Tiguan"],
    "Hyundai": ["Elantra", "Tucson", "Sonata"],
    "Kia": ["Forte", "Sorento", "Sportage"],
    "Subaru": ["Outback", "Forester", "Impreza"],
    "Mazda": ["CX-5", "Mazda3", "CX-9"],
    "Lexus": ["RX", "ES", "IS"],
    "Jeep": ["Wrangler", "Grand Cherokee", "Cherokee"],
    "Ram": ["1500", "2500", "3500"],
    "GMC": ["Sierra", "Yukon", "Terrain"],
    "Dodge": ["Charger", "Challenger", "Durango"],
    "Cadillac": ["Escalade", "CT5", "XT5"],
    "Tesla": ["Model 3", "Model Y", "Model S"],
    "Porsche": ["911", "Cayenne", "Macan"],
    "Volvo": ["XC90", "XC60", "S60"],
    "Acura": ["MDX", "RDX", "TLX"],
    "Infiniti": ["QX60", "Q50", "QX80"]
}

COLORS = ["red", "blue", "white", "black", "silver", "gray"]

# === Pre-encode (and normalize) all text features ===
with torch.no_grad():
    make_features = F.normalize(model.encode_text(tokenizer([f"a {m} car" for m in MAKES]).to(device)), dim=-1)
    color_features = F.normalize(model.encode_text(tokenizer([f"a {c} car" for c in COLORS]).to(device)), dim=-1)
    model_text_features = {
        make: F.normalize(model.encode_text(tokenizer([f"{make} {mod}" for mod in models]).to(device)), dim=-1)
        for make, models in TOP3_MODELS.items()
    }

# === Main classifier ===
def what_is_this_car(car_image):
    try:
        pil_image = Image.fromarray(cv2.cvtColor(car_image, cv2.COLOR_BGR2RGB))
        image_input = preprocess(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = F.normalize(model.encode_image(image_input), dim=-1)

        # === MAKE ===
        make_sims = (image_features @ make_features.T).squeeze(0)
        make_idx = torch.argmax(make_sims).item()
        top_make = MAKES[make_idx]
        make_conf = make_sims[make_idx].item()

        # === MODEL ===
        if top_make in model_text_features:
            model_features = model_text_features[top_make]
            model_sims = (image_features @ model_features.T).squeeze(0)
            model_idx = torch.argmax(model_sims).item()
            top_model = TOP3_MODELS[top_make][model_idx]
            model_conf = model_sims[model_idx].item()
        else:
            top_model = "Unknown"
            model_conf = 0.0

        # === COLOR ===
        color_sims = (image_features @ color_features.T).squeeze(0)
        color_idx = torch.argmax(color_sims).item()
        top_color = COLORS[color_idx]

        # Map cosine sim (-1 to 1) → pseudo-confidence 0–1
        make_conf = (make_conf + 1) / 2
        model_conf = (model_conf + 1) / 2

        print(f"CLIP: {top_color} {top_make} {top_model} "
              f"(make_conf={make_conf:.2f}, model_conf={model_conf:.2f})")

        return {
            "make": top_make,
            "model": top_model,
            "color": top_color,
            "make_conf": round(make_conf, 3),
            "model_conf": round(model_conf, 3)
        }

    except Exception as e:
        print(f"CLIP Error: {e}")
        return {
            "make": "Unknown", "model": "", "color": "Unknown",
            "make_conf": 0.0, "model_conf": 0.0
        }

# === Optional threaded API ===
label_queue = queue.Queue()
results_dict = {}
results_lock = threading.Lock()

def _classify_car(track_id, car_image):
    label = what_is_this_car(car_image)
    with results_lock:
        results_dict[track_id] = label

def label_worker():
    while True:
        track_id, crop = label_queue.get()
        if crop is None:
            break
        _classify_car(track_id, crop)
        label_queue.task_done()

threading.Thread(target=label_worker, daemon=True).start()

def enqueue_for_label(track_id, crop):
    label_queue.put((track_id, crop))

def get_label(track_id):
    with results_lock:
        return results_dict.pop(track_id, None)
