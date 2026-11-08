# Nanotech Suit-Up

## Application FastAPI + WebSocket + Three.js

Une deuxième interface complète est disponible dans `frontend/` et utilise le serveur défini dans `backend/server.py`. Le navigateur capture la webcam, recadre et compresse les images JPEG, puis les envoie au WebSocket `ws://localhost:8000/ws/pose`. Le serveur traite chaque image avec OpenCV et MediaPipe Pose, puis renvoie les 33 articulations, les matrices de transformation, une échelle apparente du corps et la mesure de segmentation. Le flux vidéo reste sur la machine locale; il n’est pas publié sur Internet.

### Démarrage sous Windows

Utilisez Python 3.11 ou 3.12. MediaPipe 0.10.21 est fixé pour fournir `mediapipe.solutions.pose`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
python backend\server.py
```

Ouvrez `http://127.0.0.1:8000`, autorisez la webcam, puis placez les épaules et le bassin dans le cadre. Le bouton **INITIALISER LA CAMÉRA** ouvre le flux et se connecte au service local; l’interface affiche la pose, le squelette débogué, la latence et l’état des plaques. Le bouton d’activation lance le déploiement des plaques procédurales. Le serveur peut télécharger le modèle MediaPipe lors de sa première initialisation; une connexion Internet peut être nécessaire au premier démarrage.

Le profil rapide compresse jusqu’à 20 images/s en JPEG 512 px, avec le modèle MediaPipe de complexité 1 et la segmentation désactivée. Une seule image est traitée à la fois pour éviter une file d’attente et conserver une latence faible. Pour privilégier la précision au débit, configurez le serveur avant son démarrage :

```powershell
$env:NANOTECH_MODEL_COMPLEXITY = "2"
$env:NANOTECH_ENABLE_SEGMENTATION = "1"
python backend\server.py
```

Le débit affiché est mesuré réellement; 60 images/s ne peuvent pas être garanties sur tous les matériels. L’inférence MediaPipe, le CPU/GPU, la webcam, la compression et la latence réseau déterminent le résultat. Les coordonnées monoculaires et l’échelle apparente ne mesurent pas la profondeur réelle. Les pièces générées sont un habillage de démonstration; un GLB anatomiquement adapté et riggé est nécessaire pour obtenir une armure photoréaliste.

### Modules

- `backend/server.py` : FastAPI, endpoint de santé, WebSocket binaire, décodage OpenCV, segmentation et matrices d’articulations MediaPipe.
- `backend/requirements.txt` : versions Python prises en charge pour cette version du pipeline MediaPipe.
- `frontend/index.html` : interface Stark, vidéo, canvas 3D, télémétrie et CSS.
- `frontend/js/tracker.js` : caméra, compression JPEG, reconnexion WebSocket, contrôle de cadence et événements de pose.
- `frontend/js/app.js` : rendu Three.js, armure procédurale, lueur, affichage de pose et déploiement par particules.

L’application Vite du répertoire racine reste disponible séparément avec `npm run dev`. Elle utilise le pipeline MediaPipe exécuté dans le navigateur; l’application FastAPI se lance avec `python backend\server.py`.

Démonstrateur d’essayage AR local : webcam, modèles Pose/Face/Hand Landmarker de MediaPipe Tasks Vision, armure articulée de démonstration en Three.js, bloom et effet de dissolution shader. La scène est rendue dans le navigateur; la vidéo n’est pas téléversée. Le modèle Pose « full » détecte les articulations du corps. Le panneau affiche une estimation de confiance calculée à partir des articulations suivies; activer **AFFICHER LES ARTICULATIONS SUIVIES** pour visualiser le squelette détecté. L’armure procédurale est ajustée articulation par articulation : un poignet ou une cheville hors cadre ne masque plus tout le reste du corps. La caméra et l’armure démarrent dès que le modèle de pose est prêt; les modèles visage et mains se chargent ensuite en parallèle et sont évalués à une fréquence réduite pour préserver la fluidité. Le chargement initial peut être annulé depuis le bouton de la caméra.

Il s’agit d’une superposition visuelle à l’écran, pas d’une armure physique ni d’un vêtement reconstruit avec une précision de mesure. Une webcam RGB standard ne fournit pas la profondeur réelle; l’occlusion 3D, les proportions et la stabilité dépendent donc du cadrage, de l’éclairage, de la caméra et du modèle importé. MediaPipe Pose est le détecteur d’articulations; OpenCV seul est une bibliothèque de vision générale, pas un modèle de pose prêt à remplacer ce détecteur. Three.js positionne ensuite l’armure sur les articulations. Aucun serveur n’est nécessaire pour le traitement vidéo.

## Démarrage

```powershell
npm install
npm run dev
```

Ouvrir l’URL localhost indiquée par Vite, autoriser la webcam et se placer dans le champ de la caméra. Le téléchargement des fichiers WASM et des modèles MediaPipe s’effectue lors de la première initialisation; une connexion Internet est requise. Le paquet MediaPipe et ses fichiers WASM doivent utiliser la même version (actuellement 0.10.35). L’application tente l’accélération GPU puis le délégué CPU si le GPU n’est pas disponible. Pour arrêter la webcam, utiliser le bouton en haut de l’interface.

## Importer son armure

Choisir un fichier `.glb` avec le bouton **CHARGER UN MODÈLE 3D**. Un GLB est préférable à un GLTF car ses textures et ressources sont empaquetées avec le modèle. Le modèle doit être debout, l’axe vertical étant Y, sa base proche de Y=0, et son origine horizontale au centre du personnage. Sa hauteur et sa position sont automatiquement ajustées à la distance tête-chevilles détectée. Les clips d’animation inclus sont lus s’ils existent. Un shader de dissolution et le bloom s’appliquent aux matériaux compatibles du modèle.

Les rigs humanoïdes dont les os suivent des noms courants (par exemple `mixamorigLeftArm`, `LeftForeArm`, `RightUpLeg`, `Hips` et `Head`) sont retargetés sur les landmarks suivis. Pour un rig personnalisé, renommer les os correspondants dans Blender ou adapter `classifyBone()` dans `src/main.js`. Les modèles sans os compatibles restent cadrés et mis à l’échelle sur le corps, sans retargeting de leurs membres.

On peut aussi placer un GLB dans `public/models/` pour le conserver dans le projet, puis le choisir dans l’interface avec un import local. Des noms d’os humanoïdes et un clip de repos rendent le résultat plus naturel. Le modèle procédural fourni reste sélectionnable à tout moment. Les GLB compatibles avec la licence du projet sont à fournir par l’utilisateur; aucun modèle Iron Man n’est inclus.

## Ajuster la pose et les proportions

Dans `src/main.js`, `pointToWorld()` convertit les coordonnées MediaPipe en espace de la scène. Les facteurs d’ajustement des dimensions de l’armure procédurale (largeur d’épaules, épaisseur du buste et largeur des membres) sont dans `updateProceduralSuit()`. `placeSegment()` ajuste les segments entre les articulations. Les seuils de visibilité pour le suit-up et sa durée (4 800 ms) sont respectivement dans `partProgress()` et `updateSuitProgress()`.

Le calcul des landmarks s’exécute seulement à l’arrivée d’une nouvelle image vidéo; le rendu WebGL, le bloom et l’animation continuent dans `requestAnimationFrame`. L’accès caméra exige `localhost` ou HTTPS. La qualité et le débit dépendent des capacités GPU, de l’éclairage, de la webcam et du navigateur; une cadence de 30–60 FPS ne peut pas être garantie sur tout matériel.
