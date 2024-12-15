import mido
from mido import Message
import time

# Constantes MIDI CC pour le tracking des mains
THUMB_INDEX_DISTANCE_CC_LEFT = 7   # Distance pouce-index main gauche
THUMB_INDEX_DISTANCE_CC_RIGHT = 8  # Distance pouce-index main droite
PINKY_LENGTH_CC_LEFT = 10         # Longueur du petit doigt gauche
PINKY_LENGTH_CC_RIGHT = 11        # Longueur du petit doigt droit
TWO_HANDS_INDEX_DISTANCE_CC = 12  # Distance entre les index des deux mains
TWO_HANDS_THUMB_DISTANCE_CC = 13  # Distance entre les pouces des deux mains

def send_hand_controls(output, left_hand_data=None, right_hand_data=None, two_hands_data=None):
    """
    Envoie les contrôles MIDI pour les mouvements des mains
    """
    if output is None:
        return
    
    messages = []
    
    if left_hand_data:
        thumb_index_dist, pinky_length = left_hand_data
        output.send(Message('control_change', control=THUMB_INDEX_DISTANCE_CC_LEFT, value=int(thumb_index_dist)))
        output.send(Message('control_change', control=PINKY_LENGTH_CC_LEFT, value=int(pinky_length)))
        messages.append(f"Left Hand (CC7={int(thumb_index_dist)}, CC10={int(pinky_length)})")
    
    if right_hand_data:
        thumb_index_dist, pinky_length = right_hand_data
        output.send(Message('control_change', control=THUMB_INDEX_DISTANCE_CC_RIGHT, value=int(thumb_index_dist)))
        output.send(Message('control_change', control=PINKY_LENGTH_CC_RIGHT, value=int(pinky_length)))
        messages.append(f"Right Hand (CC8={int(thumb_index_dist)}, CC11={int(pinky_length)})")
    
    if two_hands_data:
        index_distance, thumb_distance = two_hands_data
        output.send(Message('control_change', control=TWO_HANDS_INDEX_DISTANCE_CC, value=int(index_distance)))
        output.send(Message('control_change', control=TWO_HANDS_THUMB_DISTANCE_CC, value=int(thumb_distance)))
        messages.append(f"Two Hands (CC12={int(index_distance)}, CC13={int(thumb_distance)})")
    
    if messages:
        message = "MIDI Controls: " + " | ".join(messages)
        print(f"[{time.strftime('%H:%M:%S')}] {message}")

def send_midi_matrix(matrix, output=None, bpm=85, beg=0, end=0):
    """
    Envoie la matrice MIDI à Ableton via un port virtuel
    """
    # Vérifier si on a un port valide
    if output is None:
        output = check_midi_port()
        if output is None:
            return
        should_close = True
    else:
        should_close = False

    try:
        print(f"\n[{time.strftime('%H:%M:%S')}] Démarrage de l'envoi de la séquence MIDI...")
        
        # Démarrer l'enregistrement dans Ableton
        output.send(Message('control_change', control=116, value=127))
        output.send(Message('control_change', control=115, value=127))
        
        # Attendre avant de commencer
        time.sleep(beg)

        # Trier la matrice par temps de début
        matrix.sort(key=lambda x: x[2])

        # Variable pour suivre le temps actuel
        current_time = 0

        # Créer la liste des événements (note_on et note_off)
        all_events = []
        for note in matrix:
            pitch, velocity, start_time, duration = note
            all_events.append((start_time, 'note_on', pitch, velocity))
            all_events.append((start_time + duration, 'note_off', pitch, velocity))

        # Trier tous les événements par temps
        all_events.sort(key=lambda x: x[0])

        # Envoyer les événements
        for event_time, event_type, pitch, velocity in all_events:
            # Calculer le temps relatif
            relative_time = max(0, event_time - current_time)
            
            # Convertir le temps en secondes selon le BPM
            wait_time = (relative_time ) * (60 / bpm)
            
            # Attendre le bon moment
            time.sleep(wait_time)
            
            # Envoyer l'événement MIDI
            output.send(Message(event_type, note=pitch, velocity=velocity))
            
            # Mettre à jour le temps actuel
            current_time = event_time

        # Attendre avant de terminer
        time.sleep(end)

        # Arrêter l'enregistrement dans Ableton
        output.send(Message('control_change', control=116, value=127))
        
        # Afficher un résumé de l'envoi
        total_notes = len(matrix)
        total_events = len(all_events)
        print(f"[{time.strftime('%H:%M:%S')}] Séquence MIDI envoyée avec succès !")
        print(f"- {total_notes} notes envoyées")
        print(f"- {total_events} événements MIDI (note_on/note_off)")
        print(f"- Durée totale: {current_time/1000:.2f} secondes")

    except Exception as e:
        print(f"\n[{time.strftime('%H:%M:%S')}] Erreur lors de l'envoi MIDI: {str(e)}")
        return False
    
    finally:
        # Fermer le port MIDI seulement si on l'a ouvert nous-mêmes
        if should_close:
            output.close()
    
    return True  # Retourner True si l'envoi s'est bien passé

def check_midi_port(port_name="loopMIDI Port"):
    """Vérifie la disponibilité du port MIDI et tente de le réinitialiser si nécessaire"""
    # Ajouter "1" au nom du port
    port_name = port_name + " 1"
    
    try:
        available_ports = mido.get_output_names()
        virtual_port = next((port for port in available_ports if port == port_name), None)
        
        if virtual_port is None:
            print(f"\nErreur: Port '{port_name}' non trouvé")
            print("Ports disponibles:", available_ports)
            print("\nVérifiez que:")
            print("1. loopMIDI est installé")
            print(f"2. Un port virtuel nommé exactement '{port_name}' est créé dans loopMIDI")
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
            print(f"2. Supprimer et recréer le port virtuel avec le nom exact '{port_name}'")
            print("3. Redémarrer les applications utilisant MIDI")
            return None
            
    except Exception as e:
        print(f"\nErreur lors de la recherche des ports MIDI: {str(e)}")
        return None

if __name__ == "__main__":
    # Test avec une matrice simple
    test_matrix = [
        [60, 100, 0, 1],    # Do, vélocité 100, temps 0, durée 1
        [64, 100, 1, 1],    # Mi, vélocité 100, temps 1, durée 1
        [67, 100, 2, 1],    # Sol, vélocité 100, temps 2, durée 1
    ]
    send_midi_matrix(test_matrix) 