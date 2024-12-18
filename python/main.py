import os
import cv2
import mido

from camera_detection import main as camera_main

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    try:
        import mediapipe
        import numpy
        import mido
        import cv2
        return True
    except ImportError as e:
        print(f"Erreur: Dépendance manquante - {str(e)}")
        print("Veuillez installer toutes les dépendances avec:")
        print("pip install opencv-python mediapipe numpy mido")
        return False

def check_video_file(video_path):
    # """Vérifie que le fichier vidéo existe"""
    # if not os.path.exists(video_path):
    #     print(f"Erreur: Le fichier vidéo '{video_path}' n'existe pas")
    #     print("Veuillez placer le fichier vidéo dans le même dossier que ce script")
    #     return False
    # return True
    """Vérifie que la webcam est disponible"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur: Impossible d'accéder à la webcam")
        print("Veuillez vérifier que:")
        print("1. Une webcam est connectée")
        print("2. Aucune autre application n'utilise la webcam")
        cap.release()
        return False
    cap.release()
    return True

def check_midi_port(virtual_port_name):
    """Vérifie qu'un port loopMIDI est disponible"""
    available_ports = mido.get_output_names()
    virtual_port = next((port for port in available_ports if port.startswith(virtual_port_name)), None)
    
    if virtual_port is None:
        print("Erreur: Aucun port loopMIDI trouvé")
        print("Ports MIDI disponibles:", available_ports)
        print("\nVeuillez:")
        print("1. Installer loopMIDI")
        print("2. Créer un port virtuel dans loopMIDI")
        print("3. Configurer Ableton Live pour utiliser ce port")
        return False
    return True

def check_camera(camera_source):
    """Vérifie que la webcam est disponible"""
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print("Erreur: Impossible d'accéder à la webcam")
        print("Veuillez vérifier que:")
        print("1. Une webcam est connectée")
        print("2. Aucune autre application n'utilise la webcam")
        cap.release()
        return False
    cap.release()
    return True

def main(virtual_port_name, video_path, camera_source, live_midi=True, bpm=85):
    """Fonction principale"""
    print("Vérification de la configuration...")
    print(f"BPM: {bpm}")
    
    # Vérifier toutes les dépendances et conditions
    checks = [
        ("Dépendances", check_dependencies()),
        ("Fichier vidéo", check_video_file(video_path)),
        ("Port MIDI", check_midi_port(virtual_port_name)),
        ("Webcam", check_camera(camera_source))
    ]
    
    # Afficher les résultats des vérifications
    all_passed = True
    print("\nRésultats des vérifications:")
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
        all_passed = all_passed and passed
    
    if not all_passed:
        print("\nCertaines vérifications ont échoué. Veuillez corriger les erreurs ci-dessus.")
        return
    
    print("\nTout est prêt !")
    print("\nCommandes disponibles:")
    if live_midi:
        print("- 'a' : Envoyer la séquence MIDI basée sur les cercles détectés")
    else:
        print("- 'a' : Créer un fichier MIDI basé sur les cercles détectés")
    print("- 'q' : Quitter le programme")
    print("\nDémarrage du programme...")
    
    try:
        # Lancer le programme principal avec les paramètres
        camera_main(webcam_source=camera_source, board_source=video_path, 
                   virtual_port_name=virtual_port_name, live_midi=live_midi, bpm=bpm)
    except KeyboardInterrupt:
        print("\nProgramme arrêté par l'utilisateur")
    except Exception as e:
        print(f"\nErreur inattendue: {str(e)}")
    finally:
        print("\nFermeture du programme")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(virtual_port_name="loopMIDI Port", 
         video_path="data/video_board.mp4", 
         camera_source=0, 
         live_midi=False,
         bpm=85)
  