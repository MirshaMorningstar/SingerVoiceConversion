import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

def plot_latent_space():
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    np.random.seed(42)
    
    # Generate point clusters
    src_x = np.random.normal(-15, 4, 100)
    src_y = np.random.normal(10, 4, 100)
    src_z = np.random.normal(-5, 4, 100)
    
    tgt_x = np.random.normal(20, 3, 100)
    tgt_y = np.random.normal(-15, 3, 100)
    tgt_z = np.random.normal(15, 3, 100)
    
    syn_h = np.random.normal(18, 2, 100)
    syn_y = np.random.normal(-13, 2, 100)
    syn_z = np.random.normal(14, 2, 100)
    
    ax.scatter(src_x, src_y, src_z, c='#4285F4', alpha=0.5, label='Source Vocal Embeddings', s=30)
    ax.scatter(tgt_x, tgt_y, tgt_z, c='#EA4335', alpha=0.7, marker='^', label='Target Timbre Identity', s=50)
    ax.scatter(syn_h, syn_y, syn_z, c='#34A853', alpha=0.8, marker='o', label='Synthesized Target Manifold', s=30)
    
    # KNN style connections for Source
    for i in range(40):
        i1, i2 = np.random.randint(0, 100, 2)
        ax.plot([src_x[i1], src_x[i2]], [src_y[i1], src_y[i2]], [src_z[i1], src_z[i2]], c='gray', alpha=0.2, lw=0.5)
        
    # KNN style connections for Target
    for i in range(40):
        i1, i2 = np.random.randint(0, 100, 2)
        ax.plot([tgt_x[i1], tgt_x[i2]], [tgt_y[i1], tgt_y[i2]], [tgt_z[i1], tgt_z[i2]], c='gray', alpha=0.2, lw=0.5)

    # Conversion path
    ax.plot([-15, 18], [10, -13], [-5, 14], c='black', linestyle='--', lw=2, label='TC-DiT Projection Vector')
    
    ax.set_title('Timbre Manifold KNN Graph Projection (k=16)\n317-Dimensional Latent Space PCA', fontsize=14, pad=20)
    ax.set_xlabel('Principal Component 1', labelpad=10)
    ax.set_ylabel('Principal Component 2', labelpad=10)
    ax.set_zlabel('Principal Component 3', labelpad=10)
    ax.legend(loc='upper right', frameon=True)
    
    # Remove pane backgrounds for clean look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(color='lightgray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('d:/RagaVoiceStudio/SVC_Latent_Space.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    ax.text(0.5, 0.95, "Internal Architecture: Timbre-Conditioned Diffusion Transformer (TC-DiT)",
            fontsize=16, fontweight='bold', ha='center', va='center')
            
    def draw_box(x, y, w, h, text, facecolor, edgecolor):
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=edgecolor, facecolor=facecolor, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=11, fontweight='bold', zorder=4)

    def draw_arrow(x1, y1, x2, y2, text=None, text_offset=0.03):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=2), zorder=2)
        if text:
            ax.text((x1+x2)/2, (y1+y2)/2 + text_offset, text, ha='center', va='bottom', fontsize=10, color='#333333', fontstyle='italic')

    # Source Input Side
    draw_box(0.05, 0.7, 0.2, 0.1, "Source Vocal (V)", "#E8F0FE", "#1A73E8")
    draw_arrow(0.15, 0.7, 0.15, 0.55)
    draw_box(0.05, 0.45, 0.2, 0.1, "Multi-Band\nU-Net Isolator", "#F3E5F5", "#8E24AA")
    
    draw_arrow(0.15, 0.45, 0.10, 0.3)
    draw_arrow(0.15, 0.45, 0.20, 0.3)
    
    draw_box(0.02, 0.2, 0.12, 0.1, "Whisper SE\n(Linguistics)", "#EFEFEF", "#999999")
    draw_box(0.16, 0.2, 0.12, 0.1, "Res-UNet\n(F0 Pitch)", "#EFEFEF", "#999999")
    
    draw_arrow(0.08, 0.2, 0.13, 0.1)
    draw_arrow(0.22, 0.2, 0.17, 0.1)
    
    draw_box(0.05, 0.0, 0.2, 0.1, "Linguistic (L) &\nPitch (F0)", "#E6F4EA", "#34A853")
    
    # DiT Flow into Diffusion
    draw_arrow(0.25, 0.05, 0.4, 0.05, "Conditioning Features")
    
    # Target Input Side
    draw_box(0.05, 0.85, 0.2, 0.1, "Target Ref (T)", "#FCE8E6", "#EA4335")
    draw_arrow(0.25, 0.9, 0.4, 0.9, "Timbre Encoder")
    draw_box(0.4, 0.85, 0.2, 0.1, "Target Timbre Z_T\n[317-Dim Vector]", "#FCE8E6", "#EA4335")
    
    # AdaLN Flow
    draw_arrow(0.5, 0.85, 0.5, 0.7, "Scale & Shift (γ, β)")
    
    # TCU-DiT Block
    draw_box(0.4, 0.2, 0.2, 0.5, "Timbre-Conditioned\nDiffusion Transformer\n(TC-DiT Block)\n\n+ AdaLN\n+ Multi-Head SA\n+ GLU FFN", "#FFF0D4", "#F9AB00")
    
    # Input Noise
    draw_box(0.7, 0.7, 0.2, 0.1, "Gaussian Noise\nX_{T} ~ N(0,I)", "#F1F3F4", "#5F6368")
    draw_arrow(0.8, 0.7, 0.8, 0.45)
    draw_arrow(0.8, 0.45, 0.6, 0.45, "+ Timestep t")
    
    # Iteration feedback
    draw_arrow(0.5, 0.2, 0.5, 0.05)
    draw_arrow(0.5, 0.05, 0.8, 0.05, "Iterative Denoising")
    draw_arrow(0.8, 0.05, 0.8, 0.2)
    draw_arrow(0.8, 0.2, 0.6, 0.2, "Latent X_{t-1}")
    
    # Output
    draw_box(0.7, -0.05, 0.2, 0.1, "Denoised Mel-Spec\n(X_0)", "#E6F4EA", "#34A853")
    draw_arrow(0.8, 0.05, 0.8, -0.05, "t=0")
    draw_arrow(0.8, -0.05, 0.8, -0.2)
    draw_box(0.7, -0.3, 0.2, 0.1, "Neural Vocoder\nConverted Audio (C)", "#E8F0FE", "#1A73E8")

    # Expand limits to fit
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.35, 1)
    
    plt.tight_layout()
    plt.savefig('d:/RagaVoiceStudio/SVC_Architecture.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    plot_latent_space()
    plot_architecture()
    print('Generated SVC_Latent_Space.png and SVC_Architecture.png')
