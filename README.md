AcademIA

Sistema de Previsão de Desempenho Escolar

Este é um sistema para previsão de notas finais de alunos, utilizando técnicas de Machine Learning e Regressão. O projeto foi desenvolvido em Python, utilizando bibliotecas como Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn e Streamlit para a interface da aplicação.

Objetivo
O principal objetivo do AcademIA é que ele seja capaz de prever a nota final dos alunos com base nas variáveis preditoras, identificar os principais fatores que influenciam o desempenho dos alunos e fornecer recomendações personalizadas para melhoria e acompanhamento escolar dos mesmos.

Estrutura do Projeto
academIA/
│
├── train_model.py              
├── app.py                      
├── requirements.txt            
├── README.md                   
│
├── models/                     
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── data/                       
│   └── student_performance_dataset.csv
│
└── images/                     
    ├── correlation_matrix.png
    ├── target_distribution.png
    ├── predictions_analysis.png
    └── feature_importance.png

Instalação e Execução

1. Clonar o repositório
git clone https://github.com/robertoprsilva/academIA.git
cd academIA

2. Instalar dependências
pip install -r requirements.txt

3. Executar o treinamento
python train_model.py

4. Iniciar a aplicação Streamlit localmente
streamlit run app.py
