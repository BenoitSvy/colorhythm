import cv2
import numpy as np
import mediapipe as mp
from midi_signals import send_midi_matrix, send_hand_controls
from collections import deque
import time

# Configuration de la grille de points
point_grid = np.array([[[1/9,1/5],[2/9,1/5],[3/9,1/5],[4/9,1/5],[5/9,1/5],[6/9,1/5],[7/9,1/5],[8/9,1/5]],
                     [[1/9,2/5],[2/9,2/5],[3/9,2/5],[4/9,2/5],[5/9,2/5],[6/9,2/5],[7/9,2/5],[8/9,2/5]],
                     [[1/9,3/5],[2/9,3/5],[3/9,3/5],[4/9,3/5],[5/9,3/5],[6/9,3/5],[7/9,3/5],[8/9,3/5]],
                     [[1/9,4/5],[2/9,4/5],[3/9,4/5],[4/9,4/5],[5/9,4/5],[6/9,4/5],[7/9,4/5],[8/9,4/5]]])

pitch_row = {0: 60, 1: 62, 2: 63, 3: 65}

# Initialisation de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialisation de MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)

# Initialiser les variables d'état de la bouche
mouth_open = False
mouth_state = False  # Variable de basculement
MOUTH_THRESHOLD = 30  # Ajuster cette valeur pour changer la sensibilité

# Classe pour lisser les valeurs
class SmoothedValue:
    def __init__(self, smoothing=0.5):
        self.value = 0
        self.smoothing = smoothing
        
    def update(self, new_value):
        self.value = (self.value * self.smoothing + 
                     new_value * (1 - self.smoothing))
        return self.value

# Initialisation des lisseurs
left_thumb_index_smoother = SmoothedValue(smoothing=0.8)
right_thumb_index_smoother = SmoothedValue(smoothing=0.8)
left_pinky_smoother = SmoothedValue(smoothing=0.8)
right_pinky_smoother = SmoothedValue(smoothing=0.8)
two_hands_index_smoother = SmoothedValue(smoothing=0.8)
two_hands_thumb_smoother = SmoothedValue(smoothing=0.8)

# Classe pour mettre à l'échelle les valeurs avec des limites fixes
class FixedScaler:
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
    
    def scale_to_midi(self, value):
        # Tronquer la valeur aux limites min et max
        value = np.clip(value, self.min_value, self.max_value)
        
        if self.max_value == self.min_value:
            return 0
        # Mise à l'échelle linéaire entre min et max
        normalized = (value - self.min_value) / (self.max_value - self.min_value)
        return int(normalized * 127)

# Initialiser les scalers avec des valeurs min et max fixes
thumb_index_left_scaler = FixedScaler(min_value=0.05, max_value=0.3)
thumb_index_right_scaler = FixedScaler(min_value=0.05, max_value=0.3)
left_pinky_scaler = FixedScaler(min_value=0.0, max_value=0.15)
right_pinky_scaler = FixedScaler(min_value=0.0, max_value=0.15)
two_hands_index_scaler = FixedScaler(min_value=0.1, max_value=0.50)
two_hands_thumb_scaler = FixedScaler(min_value=0.1, max_value=0.50)

def draw_progress_bar(img, label, value, y_pos, color):
    """Dessine une barre de progression avec label et valeur"""
    # Adapter les dimensions à la taille de l'image
    h, w = img.shape[:2]
    bar_width = int(w * 0.8)  # 80% de la largeur
    bar_height = int(h * 0.08)  # 8% de la hauteur
    x_start = int(w * 0.1)  # 10% de marge
    
    # Calculer une taille de police adaptative plus grande
    font_scale = min(w, h) * 0.002  # Doublé par rapport à avant
    thickness = max(2, int(min(w, h) * 0.003))  # Épaisseur adaptative
    
    # Ajouter un fond sombre pour le texte
    text = f"{label}: {value:.1f}%"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_y = y_pos - int(bar_height * 0.5)  # Positionner le texte plus près de la barre
    
    # Dessiner un rectangle noir semi-transparent derrière le texte
    cv2.rectangle(img, 
                 (x_start - 5, text_y - text_height - 5),
                 (x_start + text_width + 5, text_y + 5),
                 (0, 0, 0), -1)
    
    # Dessiner le label avec une bordure blanche pour meilleure lisibilité
    cv2.putText(img, text, (x_start, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (x_start, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # Dessiner le cadre de la barre avec une bordure plus épaisse
    cv2.rectangle(img, (x_start, y_pos), (x_start + bar_width, y_pos + bar_height), 
                 (255, 255, 255), thickness)
    
    # Dessiner la barre de progression avec un dégradé
    filled_width = int(bar_width * value / 100)
    if filled_width > 0:
        # Créer un dégradé de couleur
        for i in range(filled_width):
            alpha = i / filled_width
            bar_color = tuple(int(c * (0.5 + 0.5 * alpha)) for c in color)
            cv2.line(img, 
                    (x_start + i, y_pos + 2), 
                    (x_start + i, y_pos + bar_height - 2), 
                    bar_color, 1)

def get_dominant_color(image, mask):
    """Détermine si un cercle est plutôt Rouge, Vert ou Bleu"""
    circle_region = cv2.bitwise_and(image, image, mask=mask)
    colors = cv2.mean(circle_region, mask=mask)[:3]  # [B, G, R]
    max_color_idx = np.argmax(colors)
    if max_color_idx == 0:
        return 'B', (255, 0, 0)
    elif max_color_idx == 1:
        return 'G', (0, 255, 0)
    else:
        return 'R', (0, 0, 255)

def detect_circles(frame):
    """Détecte tous les cercles et leur couleur dominante"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=100
    )
    
    detected_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            color_name, color_bgr = get_dominant_color(frame, mask)
            detected_circles.append((x, y, r, color_bgr, color_name))
    
    return detected_circles

def order_points(pts):
    """Ordonne les points dans l'ordre: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    
    return rect

def detect_board(frame):
    """Détecte le plateau de jeu (rectangle ou trapèze)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        if 4 <= len(approx) <= 6:
            if len(approx) > 4:
                hull = cv2.convexHull(approx)
                if len(hull) >= 4:
                    approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
            
            if len(approx) == 4:
                if cv2.isContourConvex(approx):
                    area = cv2.contourArea(approx)
                    if area > 1000:
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect_ratio = float(w)/h
                        if 0.5 <= aspect_ratio <= 2.0:
                            return approx.reshape(4, 2)
    return None

def get_warped_image(frame, corners):
    """Applique une transformation perspective pour redresser l'image"""
    if corners is None:
        return None
    
    rect = order_points(corners)
    width = int(np.mean([
        np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2)),
        np.sqrt(((rect[3][0] - rect[2][0]) ** 2) + ((rect[3][1] - rect[2][1]) ** 2))
    ]))
    height = int(np.mean([
        np.sqrt(((rect[3][0] - rect[0][0]) ** 2) + ((rect[3][1] - rect[0][1]) ** 2)),
        np.sqrt(((rect[2][0] - rect[1][0]) ** 2) + ((rect[2][1] - rect[1][1]) ** 2))
    ]))
    
    if width <= 0 or height <= 0:
        return None
    
    target_ratio = 3/4
    current_ratio = height / width
    min_dimension = 100
    
    if current_ratio > target_ratio:
        height = max(int(width * target_ratio), min_dimension)
    else:
        width = max(int(height / target_ratio), min_dimension)
    
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (width, height))
    
    return warped

def scale_point_grid(point_grid, scale_factor):
    """Ajuste l'échelle de la grille de points"""
    scaled_grid_copy = point_grid.copy()
    scaled_grid_copy[:, :, 0] *= scale_factor[1]
    scaled_grid_copy[:, :, 1] *= scale_factor[0]
    return scaled_grid_copy

def get_music_score(point_grid, pointcircles, scale_factor):
    """Détermine quels points de la grille sont occupés par des cercles"""
    point_grid_copy = scale_point_grid(point_grid, scale_factor)
    is_note = np.zeros((4, 8), dtype=object)  # Changé en object pour stocker la couleur

    for i, row in enumerate(point_grid_copy):
        for j, point in enumerate(row):
            for circle in pointcircles:
                x, y, r, color, color_name = circle
                if np.linalg.norm(np.array(point) - np.array([x, y])) < r:
                    is_note[i,j] = color_name  # Stocker la couleur au lieu d'un booléen
                    break
    
    return is_note

def get_midi_matrix(is_note):
    """Convertit la grille de notes en matrice MIDI avec vélocité selon la couleur"""
    midi_matrix = []
    for i, row in enumerate(is_note):
        for j, note_color in enumerate(row):
            if note_color:  # Si une note est présente
                pitch = pitch_row[i]
                # Définir la vélocité selon la couleur
                if note_color == 'R':
                    velocity = 100  # Rouge = vélocité 100
                elif note_color == 'G':
                    velocity = 50   # Vert = vélocité 50
                else:
                    velocity = 100   # Vélocité par défaut 100
                
                absolute_start_time = j  # Position horizontale = position de la noire
                duration = 1  # Durée d'une noire
                midi_matrix.append([pitch, velocity, absolute_start_time, duration])
    return midi_matrix

def print_matrix(is_note, midi_matrix):
    """Affiche la matrice de notes et la matrice MIDI dans la console"""
    print("\n=== Matrice de Notes ===")
    print("   0  1  2  3  4  5  6  7")
    for i, row in enumerate(is_note):
        print(f"{i}: {['-' if not x else x for x in row]}")
    
    print("\n=== Matrice MIDI ===")
    print("Format: [pitch, velocity, start_time, duration]")
    for note in midi_matrix:
        print(f"Note: {note}")
    print("========================\n")

def process_hands(frame, output):
    """Traite les mains et envoie les contrôles MIDI seulement si Live Modifier est ON"""
    global mouth_open, mouth_state
    
    # Créer l'image pour les mesures avec un fond noir
    h, w = frame.shape[:2]
    measurements_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Calculer les positions des barres proportionnellement à la hauteur de l'image
    y_spacing = h // 4
    y_start = y_spacing // 2
    
    # Initialiser les valeurs à 0
    left_thumb_index_value = 0
    right_thumb_index_value = 0
    index_distance_value = 0
    thumb_distance_value = 0
    
    # Variables pour les données MIDI
    left_hand_data = None
    right_hand_data = None
    two_hands_data = None
    
    # Flip the frame horizontally pour effet miroir
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Traiter le visage pour la détection de la bouche
    face_results = face_mesh.process(rgb_frame)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Dessiner les points du visage
            for idx, landmark in enumerate(face_landmarks.landmark):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
            
            # Points de la bouche (lèvre supérieure et inférieure)
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]

            # Calculer la hauteur de la bouche
            mouth_height = abs(upper_lip.y - lower_lip.y)
            
            # Vérifier si la bouche est ouverte
            if mouth_height > MOUTH_THRESHOLD/1000:
                if not mouth_open:
                    mouth_state = not mouth_state
                    mouth_open = True
            else:
                mouth_open = False

            # Afficher l'état dans le coin
            state_text = "ON" if mouth_state else "OFF"
            color = (0, 255, 0) if mouth_state else (0, 0, 255)
            cv2.putText(frame, f"Live Modifier: {state_text}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Traiter les mains
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Dessiner les points et connexions des mains
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            handedness = results.multi_handedness[hand_idx].classification[0].label
            
            # Points des doigts et calculs des distances
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
            
            # Convertir les coordonnées en pixels
            h, w, _ = frame.shape
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            pinky_tip_x, pinky_tip_y = int(pinky_tip.x * w), int(pinky_tip.y * h)
            pinky_mcp_x, pinky_mcp_y = int(pinky_mcp.x * w), int(pinky_mcp.y * h)
            
            # Dessiner la ligne entre pouce et index
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 2)
            
            # Calculer les distances
            thumb_index_dist = np.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
            pinky_length = np.sqrt((pinky_tip.x - pinky_mcp.x)**2 + (pinky_tip.y - pinky_mcp.y)**2)
            
            # Mise à jour des valeurs selon la main
            if handedness == 'Left':
                scaled_value = thumb_index_left_scaler.scale_to_midi(thumb_index_dist)
                left_thumb_index_value = min(100, (scaled_value * 100) / 127)
                pinky_value = left_pinky_scaler.scale_to_midi(pinky_length)
                left_hand_data = (scaled_value, pinky_value)
                
                # Affichage des mesures sur le frame
                cv2.putText(frame, f"L: {thumb_index_dist:.2f}", 
                           ((thumb_x + index_x) // 2, (thumb_y + index_y) // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                scaled_value = thumb_index_right_scaler.scale_to_midi(thumb_index_dist)
                right_thumb_index_value = min(100, (scaled_value * 100) / 127)
                pinky_value = right_pinky_scaler.scale_to_midi(pinky_length)
                right_hand_data = (scaled_value, pinky_value)
                
                # Affichage des mesures sur le frame
                cv2.putText(frame, f"R: {thumb_index_dist:.2f}", 
                           ((thumb_x + index_x) // 2, (thumb_y + index_y) // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Traitement des deux mains
            if len(results.multi_hand_landmarks) > 1 and hand_idx == 0:
                other_hand = results.multi_hand_landmarks[1]
                other_index = other_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                other_thumb = other_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
                
                # Convertir les coordonnées de l'autre main
                other_index_x, other_index_y = int(other_index.x * w), int(other_index.y * h)
                other_thumb_x, other_thumb_y = int(other_thumb.x * w), int(other_thumb.y * h)
                
                # Calcul des distances entre les mains
                index_distance = np.sqrt((index_tip.x - other_index.x)**2 + (index_tip.y - other_index.y)**2)
                thumb_distance = np.sqrt((thumb_tip.x - other_thumb.x)**2 + (thumb_tip.y - other_thumb.y)**2)
                
                # Dessiner les lignes entre les mains
                cv2.line(frame, (index_x, index_y), (other_index_x, other_index_y), (255, 0, 0), 2)  # Ligne bleue entre les index
                cv2.line(frame, (thumb_x, thumb_y), (other_thumb_x, other_thumb_y), (0, 0, 255), 2)  # Ligne rouge entre les pouces
                
                # Afficher les distances
                index_mid_x = (index_x + other_index_x) // 2
                index_mid_y = (index_y + other_index_y) // 2
                thumb_mid_x = (thumb_x + other_thumb_x) // 2
                thumb_mid_y = (thumb_y + other_thumb_y) // 2
                
                cv2.putText(frame, f"I: {index_distance:.2f}", (index_mid_x, index_mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f"T: {thumb_distance:.2f}", (thumb_mid_x, thumb_mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Mise à jour des valeurs pour les deux mains
                index_value = two_hands_index_scaler.scale_to_midi(index_distance)
                thumb_value = two_hands_thumb_scaler.scale_to_midi(thumb_distance)
                index_distance_value = min(100, (index_value * 100) / 127)
                thumb_distance_value = min(100, (thumb_value * 100) / 127)
                two_hands_data = (index_value, thumb_value)
    
    # Envoyer les données MIDI seulement si Live Modifier est ON
    if mouth_state and (left_hand_data is not None or right_hand_data is not None or two_hands_data is not None):
        send_hand_controls(output, left_hand_data, right_hand_data, two_hands_data)
    
    # Toujours afficher les barres de progression, indépendamment du Live Modifier
    draw_progress_bar(measurements_img, "Left Hand Thumb-Index", left_thumb_index_value, y_start, (0, 255, 0))
    draw_progress_bar(measurements_img, "Right Hand Thumb-Index", right_thumb_index_value, y_start + y_spacing, (0, 255, 0))
    draw_progress_bar(measurements_img, "Two Hands Index Distance", index_distance_value, y_start + 2*y_spacing, (255, 0, 0))
    draw_progress_bar(measurements_img, "Two Hands Thumb Distance", thumb_distance_value, y_start + 3*y_spacing, (0, 0, 255))
    
    return frame, measurements_img

def arrange_windows():
    """Crée une fenêtre principale contenant toutes les autres fenêtres"""
    # Taille de la fenêtre principale (16:9)
    main_width = 1920
    main_height = 1080
    
    # Créer l'image de fond noire
    main_frame = np.zeros((main_height, main_width, 3), dtype=np.uint8)
    
    # Calculer les dimensions pour chaque sous-fenêtre
    sub_width = main_width // 2
    sub_height = main_height // 2
    
    # Calculer les tailles en conservant les ratios
    # Hand Tracking (4:3)
    ht_height = min(sub_height - 40, int((sub_width - 40) * 0.75))  # ratio 4:3
    ht_width = int(ht_height * 4/3)
    
    # Board Detection (4:3)
    bd_height = ht_height  # même taille que Hand Tracking
    bd_width = ht_width
    
    # Warped Board (1:1)
    wb_size = min(sub_width - 40, sub_height - 40)  # carré
    
    # Measurements (2:3)
    meas_width = min(sub_width - 40, int((sub_height - 40) * 2/3))
    meas_height = int(meas_width * 3/2)
    
    # Calculer les positions pour centrer dans chaque quart
    ht_x = (sub_width - ht_width) // 2
    ht_y = (sub_height - ht_height) // 2
    
    bd_x = sub_width + (sub_width - bd_width) // 2
    bd_y = (sub_height - bd_height) // 2
    
    # Échanger les positions de warped et measurements
    meas_x = (sub_width - meas_width) // 2
    meas_y = sub_height + (sub_height - meas_height) // 2
    
    wb_x = sub_width + (sub_width - wb_size) // 2
    wb_y = sub_height + (sub_height - wb_size) // 2
    
    return {
        'main_frame': main_frame,
        'positions': {
            'hand_tracking': ((ht_x, ht_y), (ht_width, ht_height)),
            'board_detection': ((bd_x, bd_y), (bd_width, bd_height)),
            'warped_board': ((wb_x, wb_y), (wb_size, wb_size)),
            'measurements': ((meas_x, meas_y), (meas_width, meas_height))
        }
    }

def main(webcam_source, board_source, virtual_port_name):
    """Fonction principale pour la détection et le traitement
    Args:
        webcam_source: Source de la webcam (défaut: 0)
        board_source: Source de la vidéo du plateau (défaut: "project/vid3.mp4")
        virtual_port_name: Nom du port MIDI virtuel (défaut: "loopMIDI Port")
    """
    # Initialisation des deux sources vidéo
    hand_cap = cv2.VideoCapture(webcam_source)  # Source pour les mains
    board_cap = cv2.VideoCapture(board_source)  # Source pour le plateau
    
    if not hand_cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la source webcam {webcam_source}")
        return
    if not board_cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la source vidéo {board_source}")
        return

    # Configurer la vidéo pour une lecture plus fluide
    board_cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    
    # Utiliser un délai fixe plus court pour la lecture
    delay = 1  # 1ms de délai minimum

    # Initialiser la connexion MIDI
    import mido
    available_ports = mido.get_output_names()
    virtual_port = next((port for port in available_ports if port.startswith(virtual_port_name)), None)
    
    if virtual_port is None:
        print(f"Erreur: Port {virtual_port_name} non trouvé")
        print("Ports disponibles:", available_ports)
        return

    output = mido.open_output(virtual_port)
    print(f"Connecté au port: {virtual_port}")

    # Créer la fenêtre principale
    cv2.namedWindow('Chief Orchestra', cv2.WINDOW_NORMAL)
    
    # Obtenir la configuration des fenêtres
    window_config = arrange_windows()
    main_frame = window_config['main_frame']
    positions = window_config['positions']
    
    # Variables pour stocker les dernières détections valides
    last_valid_corners = None
    last_valid_warped = None
    last_valid_circles = None

    try:
        while hand_cap.isOpened():
            # Créer une nouvelle image de fond noire pour chaque frame
            display_frame = main_frame.copy()
            
            # Lecture des deux sources
            hand_ret, hand_frame = hand_cap.read()
            board_ret, board_frame = board_cap.read()

            if not hand_ret:
                print("Erreur: Impossible de lire la webcam")
                break

            # Gérer la lecture en boucle de la vidéo du plateau
            if not board_ret:
                board_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                board_ret, board_frame = board_cap.read()
                if not board_ret:
                    continue

            # Traitement des mains pour les contrôles MIDI
            hand_frame, measurements_img = process_hands(hand_frame, output)
            
            # Redimensionner et placer Hand Tracking
            pos, size = positions['hand_tracking']
            hand_frame_resized = cv2.resize(hand_frame, size)
            display_frame[pos[1]:pos[1]+size[1], pos[0]:pos[0]+size[0]] = hand_frame_resized

            # Toujours afficher la vidéo du plateau
            pos, size = positions['board_detection']
            board_frame_resized = cv2.resize(board_frame, size)
            display_frame[pos[1]:pos[1]+size[1], pos[0]:pos[0]+size[0]] = board_frame_resized

            # Détection du plateau et des cercles (sans affecter l'affichage)
            board_corners = detect_board(board_frame.copy())
            if board_corners is not None:
                last_valid_corners = board_corners
                warped = get_warped_image(board_frame.copy(), board_corners)
                
                if warped is not None:
                    last_valid_warped = warped.copy()
                    circles = detect_circles(warped)
                    if circles:
                        last_valid_circles = circles

            # Afficher la dernière détection valide dans Warped Board
            if last_valid_warped is not None:
                warped_with_detections = last_valid_warped.copy()
                
                scaled_point_grid = scale_point_grid(point_grid, (warped_with_detections.shape[0], warped_with_detections.shape[1]))
                for i, row in enumerate(scaled_point_grid):
                    for j, point in enumerate(row):
                        cv2.circle(warped_with_detections, tuple(point.astype(int)), 1, (0, 0, 255), -1)

                if last_valid_circles:
                    for (x, y, r, color, color_name) in last_valid_circles:
                        cv2.circle(warped_with_detections, (x, y), r, color, 2)
                        cv2.circle(warped_with_detections, (x, y), 2, color, 3)
                        cv2.putText(warped_with_detections, color_name, 
                                  (x - 10, y - r - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                pos, size = positions['warped_board']
                warped_resized = cv2.resize(warped_with_detections, (size[0], size[0]))
                display_frame[pos[1]:pos[1]+size[0], pos[0]:pos[0]+size[0]] = warped_resized

            # Redimensionner et placer la fenêtre des mesures
            pos, size = positions['measurements']
            measurements_resized = cv2.resize(measurements_img, size)
            display_frame[pos[1]:pos[1]+size[1], pos[0]:pos[0]+size[0]] = measurements_resized

            # Afficher la fenêtre principale
            cv2.imshow('Chief Orchestra', display_frame)
            
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a') and last_valid_warped is not None and last_valid_circles:
                try:
                    is_note = get_music_score(point_grid, last_valid_circles, 
                                           (last_valid_warped.shape[0], last_valid_warped.shape[1]))
                    midi_matrix = get_midi_matrix(is_note)
                    print(f"\n[{time.strftime('%H:%M:%S')}] Circle Detection: {len(midi_matrix)} notes detected")
                    print_matrix(is_note, midi_matrix)
                    if not send_midi_matrix(midi_matrix, output=output):
                        print(f"[{time.strftime('%H:%M:%S')}] Échec de l'envoi de la séquence MIDI")
                except Exception as e:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Error: {str(e)}")

    finally:
        hand_cap.release()
        board_cap.release()
        cv2.destroyAllWindows()
        output.close()

if __name__ == "__main__":
    main() 