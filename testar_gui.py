import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
# Imports de QWidget e Layouts
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QLabel, QMessageBox, QDialog, QDialogButtonBox,
                             QStackedWidget)
# Imports de QGeral e QGraficos (limpos)
from PyQt5.QtCore import Qt, QPoint, QBuffer, QIODevice
from PyQt5.QtGui import QPixmap, QIcon, QPainter, QPen, QImage
# Import para conversão de imagem em memória
from io import BytesIO 

# Estilo de Tema
STYLESHEET_LIGHT = """
    QWidget { background-color: #f0f0f0; color: #2b2b2b; }
    QPushButton { 
        background-color: #e0e0e0; border: 1px solid #b0b0b0; 
        padding: 8px; border-radius: 4px;
        font-size: 14px;
    }
    QPushButton:hover { background-color: #d0d0d0; }
    QLabel { color: #2b2b2b; font-size: 16px; }
    QDialog { background-color: #f0f0f0; }
"""

# Arquitetura do Modelo
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 1)
        
    def forward(self, x):
        return self.fc(x)

# Carregar o Modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Perceptron().to(device)

MODEL_FILE = 'perceptron_A_v5.pth'
try:
    model.load_state_dict(torch.load(MODEL_FILE))
except FileNotFoundError:
    print(f"ERRO: Arquivo '{MODEL_FILE}' não encontrado.")
    print("Por favor, rode o script 'treinar_modelo.py' primeiro.")
    sys.exit()
    
model.eval() 

# Otimizador e Custo (feedback)
optimizer = optim.SGD(model.parameters(), lr=0.01) 
criterion = nn.BCEWithLogitsLoss() 

# Transformações da Imagem
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.Resize((28, 28)),                 
    transforms.ToTensor(),                       
    transforms.Lambda(lambda x: 1.0 - x), 
    transforms.Lambda(lambda x: x.view(-1)) 
])

# Janela de Feedback (COM BOTÕES ESTILIZADOS)
class FeedbackDialog(QDialog):
    def __init__(self, title, message, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setLayout(QVBoxLayout())
        
        self.message_label = QLabel(message)
        self.message_label.setAlignment(Qt.AlignCenter)
        self.layout().addWidget(self.message_label)
        
        button_box = QDialogButtonBox()
        # Define os botões
        self.btn_certo = button_box.addButton("Ele Acertou!", QDialogButtonBox.YesRole) # YesRole = Esquerda
        self.btn_errado = button_box.addButton("Ele Errou!", QDialogButtonBox.NoRole)   # NoRole = Direita
        
        # --- APLICA OS ESTILOS PERSONALIZADOS ---
        self.btn_certo.setStyleSheet("""
            QPushButton {
                background-color: #28a745; /* Verde */
                color: white;
                border-radius: 4px; padding: 8px; font-size: 14px;
            }
            QPushButton:hover { background-color: #218838; }
        """)
        
        self.btn_errado.setStyleSheet("""
            QPushButton {
                background-color: #dc3545; /* Vermelho */
                color: white;
                border-radius: 4px; padding: 8px; font-size: 14px;
            }
            QPushButton:hover { background-color: #c82333; }
        """)
        # --- FIM DOS ESTILOS ---

        self.layout().addWidget(button_box)
        
        self.btn_certo.clicked.connect(self.on_certo)
        self.btn_errado.clicked.connect(self.on_errado)
        
        self.feedback = None 

    def on_certo(self):
        self.feedback = 'certo'
        self.accept() 

    def on_errado(self):
        self.feedback = 'errado'
        self.accept()

# Página de Desenho
class DrawingPage(QWidget):
    # (Esta classe estava limpa e não precisou de mudanças)
    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app 
        
        self.canvas_size = 280 
        self.canvas_label = QLabel()
        self.pixmap = QPixmap(self.canvas_size, self.canvas_size)
        self.clear_canvas() 
        self.canvas_label.setPixmap(self.pixmap)
        self.canvas_label.setAlignment(Qt.AlignCenter)
        
        # Layout Superior (Botão de Tema Removido)
        top_layout = QHBoxLayout()
        self.btn_voltar = QPushButton("<- Voltar")
        self.btn_voltar.clicked.connect(self.parent_app.switch_to_home) 
        self.btn_voltar.setFixedWidth(100)
        
        top_layout.addWidget(self.btn_voltar)
        top_layout.addStretch()
        
        # Layout Inferior
        bottom_layout = QHBoxLayout()
        self.btn_limpar_canvas = QPushButton("Limpar Tela")
        self.btn_limpar_canvas.clicked.connect(self.clear_canvas)
        self.btn_testar = QPushButton("Testar Desenho")
        self.btn_testar.clicked.connect(self.test_drawing)
        bottom_layout.addWidget(self.btn_limpar_canvas)
        bottom_layout.addWidget(self.btn_testar)
        
        # Layout Principal da Página
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.canvas_label, 1, Qt.AlignCenter) 
        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)
        
        self.last_point = QPoint()
        self.drawing = False

    def clear_canvas(self):
        self.pixmap.fill(Qt.white) 
        self.canvas_label.setPixmap(self.pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos() - QPoint(int((self.width() - self.canvas_size)/2), int((self.height() - self.canvas_size)/2))

    def mouseMoveEvent(self, event):
        if self.drawing and event.buttons() & Qt.LeftButton:
            painter = QPainter(self.pixmap)
            painter.setPen(QPen(Qt.black, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            
            new_point = event.pos() - QPoint(int((self.width() - self.canvas_size)/2), int((self.height() - self.canvas_size)/2))
            painter.drawLine(self.last_point, new_point)
            
            self.last_point = new_point
            self.canvas_label.setPixmap(self.pixmap) 
            painter.end()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def test_drawing(self):
        qimage = self.pixmap.toImage()
        buffer = QBuffer()
        buffer.open(QIODevice.ReadWrite)
        qimage.save(buffer, "PNG") 
        image_data = buffer.data()
        pil_img = Image.open(BytesIO(image_data)).convert('L') 
        
        try:
            image_tensor = data_transform(pil_img).unsqueeze(0).to(device)
            self.parent_app.run_test(image_tensor) 
        except Exception as e:
            QMessageBox.critical(self, 'Erro', f'Não foi possível processar o desenho: {e}')

# Página Inicial (COM LÓGICA DE BOTÃO SIMPLIFICADA)
class HomePage(QWidget):
    def __init__(self, parent_app):
        super().__init__()
        self.parent_app = parent_app 
        self.image_tensor = None
        
        # Layout Superior (Botão de Tema Removido)
        top_layout = QHBoxLayout()
        self.btn_clear = QPushButton("Limpar Imagem")
        self.btn_clear.clicked.connect(self.reset_ui)
        self.btn_clear.setFixedWidth(130)
        self.btn_clear.setVisible(False) 
        
        top_layout.addWidget(self.btn_clear)
        top_layout.addStretch() 

        # Layout Central
        center_layout = QVBoxLayout()
        center_layout.addStretch() 
        
        self.image_label = QLabel('Nenhuma imagem carregada.', self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(150)
        center_layout.addWidget(self.image_label)
        
        # Layout dos Botões
        button_layout = QHBoxLayout()
        button_layout.addStretch() 
        
        # --- LÓGICA DE BOTÃO SIMPLIFICADA ---
        self.btn_action = QPushButton('Procurar Imagem...', self)
        self.btn_action.clicked.connect(self.on_action_click) # Conecta a uma única função
        self.btn_action.setMinimumHeight(40)
        
        self.btn_draw = QPushButton('Desenhar...', self)
        self.btn_draw.clicked.connect(self.parent_app.switch_to_drawing) 
        self.btn_draw.setMinimumHeight(40)
        button_layout.addWidget(self.btn_action)
        button_layout.addWidget(self.btn_draw)
        button_layout.addStretch() 
        
        center_layout.addLayout(button_layout) 
        center_layout.addStretch() 
        
        # Layout Principal da Página
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(center_layout)
        self.setLayout(main_layout)

    def reset_ui(self):
        """Reseta a interface para o estado inicial."""
        self.image_tensor = None
        self.image_label.setText('Nenhuma imagem carregada.')
        self.image_label.setPixmap(QPixmap())
        self.btn_action.setText('Procurar Imagem...')
        self.btn_clear.setVisible(False) 

    def on_action_click(self):
        """
        Esta ÚNICA função lida com "Procurar Imagem" e "Testar Imagem",
        verificando o texto do botão para decidir o que fazer.
        """
        if self.btn_action.text() == 'Procurar Imagem...':
            # --- Modo: Procurar Imagem ---
            file_path, _ = QFileDialog.getOpenFileName(self, "Selecionar Imagem", "", 
                                                      "Imagens (*.png *.jpg *.jpeg *.bmp)")
            if file_path:
                try:
                    image = Image.open(file_path)
                    self.image_tensor = data_transform(image).unsqueeze(0).to(device)
                    pixmap = QPixmap(file_path)
                    self.image_label.setPixmap(pixmap.scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    
                    # Muda o estado do botão
                    self.btn_action.setText('Testar Imagem')
                    self.btn_clear.setVisible(True) 
                    
                except Exception as e:
                    QMessageBox.critical(self, 'Erro', f'Não foi possível processar a imagem: {e}')
                    self.reset_ui()
        
        else:
            # --- Modo: Testar Imagem ---
            if self.image_tensor is not None:
                self.parent_app.run_test(self.image_tensor)
            self.reset_ui() # Reseta a UI depois do teste

# Janela Principal
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Reconhecedor da Letra A')
        
        # Define o ícone da janela
        try:
            self.setWindowIcon(QIcon('logo_a.ico'))
        except Exception as e:
            print(f"Aviso: Não foi possível carregar 'logo_a.ico'. {e}")

        self.resize(500, 400) 
        
        self.stack = QStackedWidget()
        self.home_page = HomePage(self)
        self.drawing_page = DrawingPage(self)
        
        self.stack.addWidget(self.home_page)     
        self.stack.addWidget(self.drawing_page)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.stack)
        self.setLayout(layout)
        
        # Aplica o tema claro
        self.setStyleSheet(STYLESHEET_LIGHT)
        # Garante que o tema seja aplicado em todas as páginas
        self.home_page.setStyleSheet(STYLESHEET_LIGHT)
        self.drawing_page.setStyleSheet(STYLESHEET_LIGHT)
        self.show()

    def switch_to_home(self):
        self.stack.setCurrentIndex(0) 

    def switch_to_drawing(self):
        self.stack.setCurrentIndex(1) 

    # FUNÇÃO CENTRAL
    def run_test(self, image_tensor_to_test):
        try:
            with torch.no_grad():
                output_raw = model(image_tensor_to_test).squeeze()
                probabilidade = torch.sigmoid(output_raw).item()
                
            predicao_modelo = probabilidade > 0.5 
            
            if predicao_modelo:
                msg = f'É a letra A!\n(Confiança: {probabilidade*100:.2f}%)'
                dialog = FeedbackDialog('Resultado', msg, self)
            else:
                msg = f"NÃO é a letra A.\n(Confiança de ser 'A': {probabilidade*100:.2f}%)"
                dialog = FeedbackDialog('Resultado', msg, self)
            
            # Aplica o tema claro ao diálogo também
            dialog.setStyleSheet(STYLESHEET_LIGHT) 
            dialog.exec_() 
            
            # LÓGICA DE APRENDIZADO ATUALIZADA
            if dialog.feedback == 'errado':
                # Se errou, inverte a predição e mostra a mensagem de agradecimento
                label_correta = 1.0 if not predicao_modelo else 0.0 
                self.aprender_com_feedback(image_tensor_to_test, label_correta, show_thank_you_message=True)
            elif dialog.feedback == 'certo':
                # Se acertou, confirma a predição e aprende, mas sem mensagem
                label_correta = 1.0 if predicao_modelo else 0.0
                self.aprender_com_feedback(image_tensor_to_test, label_correta, show_thank_you_message=False)
            
        except Exception as e:
            QMessageBox.critical(self, 'Erro', f'Falha no teste: {e}')

    # Função de Aprendizado (Atualizada com Mensagem Condicional)
    def aprender_com_feedback(self, image_tensor, label_correta, show_thank_you_message):
        print(f"Aprendendo com feedback... Correto era: {label_correta}")
        
        model.train() 
        optimizer.zero_grad()
        label_tensor = torch.tensor(label_correta).float().to(device)
        output = model(image_tensor).squeeze()
        loss = criterion(output, label_tensor)
        loss.backward()
        optimizer.step()
        model.eval() 
        
        try:
            torch.save(model.state_dict(), MODEL_FILE)
            # Exibe a mensagem de agradecimento apenas se for um erro corrigido
            if show_thank_you_message:
                QMessageBox.information(self, 'Aprendizado', 
                    f'Obrigado! O modelo foi atualizado e salvo em\n{MODEL_FILE}')
        except Exception as e:
            QMessageBox.warning(self, 'Erro', f'Não foi possível salvar o modelo: {e}')

# Executar a Aplicação
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())