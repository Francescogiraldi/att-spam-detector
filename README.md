# 📱 AT&T Spam Detector - Application Streamlit

Une application web interactive pour la détection automatique de SMS indésirables utilisant des modèles d'intelligence artificielle avancés.

## 🎯 Fonctionnalités

- **Détection en temps réel** : Analysez instantanément vos SMS
- **Deux modèles IA** : Modèle de base (Neural Network) et modèle avancé (BERT)
- **Interface intuitive** : Design moderne et expérience utilisateur optimisée
- **Visualisations interactives** : Graphiques, métriques et analyses détaillées
- **Analyse comparative** : Comparaison des performances des modèles

## 🚀 Installation et lancement

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation

1. **Clonez ou téléchargez le projet**
   ```bash
   cd "PROJETS VERSIONES FINALES/PROJET_AT&T"
   ```

2. **Installez les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

3. **Lancez l'application**
   ```bash
   streamlit run app.py
   ```

4. **Ouvrez votre navigateur**
   L'application sera accessible à l'adresse : `http://localhost:8501`

## 📋 Structure du projet

```
PROJET_AT&T/
├── app.py                          # Application Streamlit principale
├── requirements.txt                # Dépendances Python
├── README.md                      # Documentation
├── AT&T_Spam_Detector.ipynb      # Notebook Jupyter original
└── spam.csv                       # Dataset (si disponible)
```

## 🔧 Utilisation

### Navigation
L'application est organisée en 5 sections principales :

1. **🏠 Accueil** : Vue d'ensemble du projet et statistiques
2. **🔍 Détection** : Interface de test pour analyser vos SMS
3. **📊 Analyse des données** : Exploration et visualisation du dataset
4. **📈 Métriques** : Performances et comparaison des modèles
5. **ℹ️ À propos** : Informations détaillées sur le projet

### Test de détection
1. Accédez à la section "🔍 Détection"
2. Choisissez le modèle (BERT ou Neural Network)
3. Saisissez votre message SMS ou utilisez les exemples
4. Consultez le résultat et l'analyse détaillée

## 🤖 Modèles d'IA

### Modèle de base (Neural Network)
- Architecture : Réseau de neurones avec embeddings
- Couches : Embedding → GlobalAveragePooling1D → Dense
- Vocabulaire : 5,000 mots les plus fréquents
- Séquence maximale : 50 tokens

### Modèle avancé (BERT)
- Architecture : DistilBERT (version allégée de BERT)
- Technique : Transfer Learning avec fine-tuning
- Paramètres : ~66 millions
- Séquence maximale : 64 tokens

## 📊 Performances

| Modèle | Précision | Rappel | F1-Score | Accuracy |
|--------|-----------|--------|----------|---------|
| Neural Network | 95.2% | 94.3% | 94.7% | 94.8% |
| BERT (DistilBERT) | 98.5% | 96.8% | 97.6% | 98.2% |

## 🛠️ Technologies utilisées

- **Frontend** : Streamlit
- **Machine Learning** : TensorFlow, PyTorch
- **NLP** : Transformers (Hugging Face), NLTK
- **Visualisation** : Plotly, Matplotlib, WordCloud
- **Data Science** : Pandas, NumPy, scikit-learn

## 📝 Notes importantes

- **Mode démonstration** : L'application fonctionne actuellement en mode simulation
- **Modèles pré-entraînés** : Pour une utilisation en production, les modèles doivent être entraînés sur le dataset complet
- **Performance** : Les métriques affichées sont basées sur les résultats du notebook original

## 🔮 Améliorations futures

- [ ] Intégration des modèles réellement entraînés
- [ ] Support de plusieurs langues
- [ ] API REST pour intégration externe
- [ ] Système de feedback utilisateur
- [ ] Détection de phishing avancée
- [ ] Analyse de sentiment

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Signaler des bugs
- Proposer de nouvelles fonctionnalités
- Améliorer la documentation
- Optimiser les performances

## 📄 Licence

Ce projet est développé à des fins éducatives et de démonstration.

---

**Développé avec ❤️ pour la cybersécurité et la protection des utilisateurs**