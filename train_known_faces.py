import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

known_dir = "known"  # Folder with subfolders for each person
known_faces = {}

for person_name in os.listdir(known_dir):
    person_folder = os.path.join(known_dir, person_name)
    if not os.path.isdir(person_folder):
        continue
    embeddings = []
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        try:
            img = Image.open(img_path)
            img = img.convert('RGB')  # Ensure 3 channels
        except Exception as e:
            print(f"Failed to open {img_path}: {e}")
            continue
        face = mtcnn(img)
        if face is not None:
            emb = resnet(face.unsqueeze(0)).detach()
            embeddings.append(emb)
    if embeddings:
        # Average embedding for the person
        known_faces[person_name] = torch.stack(embeddings).mean(0)
        print(f"Trained {person_name} with {len(embeddings)} images.")
    else:
        print(f"No faces found for {person_name}.")

torch.save(known_faces, "known_faces.pt")
print("Saved known faces embeddings to known_faces.pt")
