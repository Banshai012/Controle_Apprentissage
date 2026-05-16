import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import torch.nn as nn
import torch.nn.functional as F

# --- REDÉFINITION DU MODÈLE (Doit être identique à agent.py) ---
class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(LinearQNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear3 = nn.Linear(hidden_size1, hidden_size2)
        self.linear2 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear3(x))
        x = self.linear2(x)
        return x

def test_model(model_path):
    # 1. Chargement du modèle avec la nouvelle architecture (12 entrées)
    model = LinearQNet(12, 256, 256, 4)
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Modèle {model_path} chargé avec succès.")
    except Exception as e:
        print(f"Erreur de chargement : {e}")
        return

    model.eval()

    # 2. Configuration de l'interface graphique
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, bottom=0.4)
    
    actions = ['HAUT', 'DROITE', 'BAS', 'GAUCHE']
    x_pos = np.arange(len(actions))
    bars = ax.bar(x_pos, [0, 0, 0, 0], color='skyblue')
    
    ax.set_ylim(-20, 50) # Ajusté car avec 2 couches, les Q-values peuvent être plus hautes
    ax.set_xticks(x_pos)
    ax.set_xticklabels(actions)
    ax.set_title(f"Analyse des Q-Values (12 inputs) : {model_path}")

    # 3. Sliders pour les variables continues
    ax_debit = plt.axes([0.15, 0.25, 0.3, 0.03])
    ax_grad  = plt.axes([0.15, 0.20, 0.3, 0.03])
    ax_prev  = plt.axes([0.15, 0.15, 0.3, 0.03])
    ax_time  = plt.axes([0.15, 0.10, 0.3, 0.03])

    s_debit = Slider(ax_debit, 'Débit Norm', 0.0, 1.0, valinit=0.5)
    s_grad  = Slider(ax_grad, 'Gradient', -1.0, 1.0, valinit=0.0)
    s_prev  = Slider(ax_prev, 'Grad Prev', -1.0, 1.0, valinit=0.0)
    s_time  = Slider(ax_time, 'Temps Remt', 0.0, 1.0, valinit=1.0)

    # 4. Sélecteurs pour les Directions (One-Hot)
    ax_dir_act = plt.axes([0.55, 0.15, 0.15, 0.15], facecolor='#f0f0f0')
    ax_dir_pre = plt.axes([0.75, 0.15, 0.15, 0.15], facecolor='#f0f0f0')
    
    radio_act = RadioButtons(ax_dir_act, ('Haut', 'Droite', 'Bas', 'Gauche'), active=1)
    radio_pre = RadioButtons(ax_dir_pre, ('Haut', 'Droite', 'Bas', 'Gauche'), active=1)
    
    ax_dir_act.set_title("Direction Actuelle")
    ax_dir_pre.set_title("Direction Précédente")

    def update(val):
        # Conversion des directions Radio en index
        dir_map = {'Haut': 0, 'Droite': 1, 'Bas': 2, 'Gauche': 3}
        
        # On recrée les vecteurs One-Hot
        one_hot_act = [0, 0, 0, 0]
        one_hot_act[dir_map[radio_act.value_selected]] = 1
        
        one_hot_prev = [0, 0, 0, 0]
        one_hot_prev[dir_map[radio_pre.value_selected]] = 1

        # Construction du vecteur d'entrée final (12 valeurs)
        state = [
            s_debit.val, 
            s_grad.val, 
            s_prev.val, 
            s_time.val
        ] + one_hot_act + one_hot_prev
        
        state_tensor = torch.tensor(state, dtype=torch.float)
        
        with torch.no_grad():
            prediction = model(state_tensor)
            q_values = prediction.numpy()
        
        for rect, h in zip(bars, q_values):
            rect.set_height(h)
            rect.set_color('green' if h == np.max(q_values) else 'skyblue')
            
        fig.canvas.draw_idle()

    # Liaison des événements
    s_debit.on_changed(update)
    s_grad.on_changed(update)
    s_prev.on_changed(update)
    s_time.on_changed(update)
    radio_act.on_clicked(update)
    radio_pre.on_clicked(update)

    update(None)
    plt.show()

if __name__ == "__main__":
    # Mets ici le nom de ton nouveau modèle entraîné
    test_model('./model/model_5.pth')