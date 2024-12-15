import mido
from mido import Message
import time

# Constantes MIDI CC
CONTROLS = [
    (7, "Distance pouce-index main gauche"),
    (8, "Distance pouce-index main droite"),
    (10, "Longueur du petit doigt gauche"),
    (11, "Longueur du petit doigt droit"),
    (12, "Distance entre les index des deux mains"),
    (13, "Distance entre les pouces des deux mains"),
    # Ajouter les contrôles d'enregistrement Ableton
    (114, "Ableton - Start Recording (1)"),
    (115, "Ableton - Start Recording (2)"),
    (116, "Ableton - Stop Recording")
]

def check_midi_port(port_name="loopMIDI Port"):
    """Vérifie la disponibilité du port MIDI et tente de le réinitialiser si nécessaire"""
    try:
        available_ports = mido.get_output_names()
        virtual_port = next((port for port in available_ports if port.startswith(port_name)), None)
        
        if virtual_port is None:
            print(f"\nErreur: Port '{port_name}' non trouvé")
            print("Ports disponibles:", available_ports)
            print("\nVérifiez que:")
            print("1. loopMIDI est installé")
            print("2. Un port virtuel est créé dans loopMIDI")
            print("3. Le port n'est pas utilisé par une autre application")
            return None
        
        # Tester l'ouverture du port
        try:
            output = mido.open_output(virtual_port)
            print(f"Port MIDI connecté: {virtual_port}")
            return output
        except Exception as e:
            print(f"\nErreur lors de l'ouverture du port: {str(e)}")
            print("Essayez de:")
            print("1. Fermer puis rouvrir loopMIDI")
            print("2. Supprimer et recréer le port virtuel")
            print("3. Redémarrer les applications utilisant MIDI")
            return None
            
    except Exception as e:
        print(f"\nErreur lors de la recherche des ports MIDI: {str(e)}")
        return None

def initialize_midi_controls():
    """Initialise les contrôles MIDI un par un avec confirmation utilisateur"""
    
    output = check_midi_port()
    if output is None:
        return

    try:
        for cc, description in CONTROLS:
            print(f"\n=== Signal CC{cc} : {description} ===")
            print(f"1. Sélectionnez le paramètre à contrôler dans Ableton")
            print(f"2. Cliquez sur le paramètre pour le mapper")
            input(f"3. Appuyez sur Entrée quand vous êtes prêt à recevoir le signal CC{cc}...")
            
            print("Envoi du signal...")
            # Pour les contrôles d'enregistrement, envoyer juste une impulsion
            if cc in [114, 115, 116]:
                output.send(Message('control_change', control=cc, value=127))
                time.sleep(0.1)
            else:
                # Pour les autres contrôles, envoyer l'oscillation habituelle
                for _ in range(3):
                    output.send(Message('control_change', control=cc, value=127))
                    time.sleep(0.1)
                    output.send(Message('control_change', control=cc, value=0))
                    time.sleep(0.1)
            
            print(f"Signal CC{cc} envoyé !")
            input("Appuyez sur Entrée pour passer au signal suivant...")

    except KeyboardInterrupt:
        print("\nInitialisation interrompue par l'utilisateur")
    finally:
        output.close()
        print("\nPort MIDI fermé")

if __name__ == "__main__":
    print("=== Initialisateur de Contrôles MIDI ===")
    print("Cet outil va vous aider à mapper les contrôles MIDI dans Ableton Live")
    print("\nPréparation :")
    print("1. Ouvrez Ableton Live")
    print("2. Activez le mode MIDI Map (Ctrl+M)")
    print("\nPour chaque signal :")
    print("1. Le programme annoncera le signal à mapper")
    print("2. Sélectionnez le paramètre dans Ableton")
    print("3. Appuyez sur Entrée pour envoyer le signal")
    print("4. Le paramètre sera automatiquement mappé")
    input("\nAppuyez sur Entrée pour commencer...")
    
    initialize_midi_controls() 