# OptiFrame

Défi CodeML · Santé Numérique Sans Frontières — mesurer un verre de lunettes
recyclé depuis un téléphone, au millimètre, et en déduire une monture imprimable
en 3D, verre par verre.

**Lien de l'app :** _TODO équipe — URL HTTPS publique une fois déployée._
**QR code :** affiché dans l'onglet « À propos » de l'app.

---

## En deux minutes

1. Imprimer `public/capture-sheet.pdf` sur du A4. **L'échelle n'a pas d'importance** :
   si l'imprimante réduit la page, l'app corrige.
2. Mesurer au pied à coulisse le trait de contrôle imprimé sur la feuille, et
   saisir la valeur dans l'app (onglet Capturer). Elle est retenue.
3. Poser la feuille à plat, poser le verre dans la zone grise, haut du verre vers
   le haut de la feuille.
4. Ouvrir l'app, photographier d'au-dessus en cadrant les quatre marqueurs.

C'est tout le dispositif de capture : une feuille de papier. Il se remonte en
moins de deux minutes et n'importe qui peut le réimprimer.

---

## Précision mesurée

### Sur de vraies photos, d'un vrai verre

Huit photos d'un même verre posé sur la feuille imprimée (à 84 % — l'imprimante
ne voulait pas de 100 %), sous des angles et des distances différents, verre
mesuré au pied à coulisse à 52,1 × 39,5 mm :

| Mesure | Valeur |
|---|---|
| Erreur absolue moyenne sur A et B | **0,18 mm** |
| p95 / max | 0,35 / 0,35 mm |
| Captures mesurées | **8/8** |
| Résidu de l'ajustement d'échelle | 0,11 – 0,18 mm |
| Écart-type entre les 8 prises | A 0,13 mm, B 0,19 mm |

Reproductible : `npm run harness && python3 scripts/eval_mm.py --real pictures/aruco --control-length 84`

### Sur captures rendues (géométrie, vérité exacte)

| Mesure | Valeur |
|---|---|
| Erreur absolue moyenne sur A et B (40 captures synthétiques tenues à l'écart) | **0,55 mm** |
| p95 / max | 1,13 mm / 1,84 mm |
| Captures mesurées (les autres sont refusées, pas devinées) | 34/40 |
| Résidu de l'ajustement d'échelle (4 marqueurs, 16 correspondances) | 0,13 – 0,30 mm |
| Correction systématique calibrée | **0,00 mm** — aucune |
| Mesures fausses passant tous les garde-fous | ~1 sur 40 (voir Limites) |

Reproductible : `npm run harness && python3 scripts/eval_mm.py --synthetic 40`

Vérifié aussi **dans le navigateur**, de bout en bout, sur deux captures de
démonstration (`pictures/synthetic-demo/`, vérité terrain dans `truth.json`) :

| Verre | A mesuré / vrai | B mesuré / vrai | Temps |
|---|---|---|---|
| droit | 42,4 / 42,45 mm | 32,8 / 32,33 mm | 2,8 s |
| gauche | 46,8 / 46,88 mm | 36,4 / 36,30 mm | 1,2 s |

Aucune erreur console, interface restée utilisable pendant le calcul, monture
générée et aperçu 3D affiché. Pour refaire ce test sans rien imprimer : onglet
Capturer → « Importer une photo » avec les deux fichiers de
`pictures/synthetic-demo/`.

Le dernier point mérite d'être souligné : le balayage de calibration
(`--sweep`) place l'optimum à **zéro correction**, avec un biais résiduel de
0,04 mm. Le contour tombe donc là où est le verre parce que la géométrie est
juste, pas parce qu'une constante a été ajustée jusqu'à ce que ça tombe juste.

**À compléter avant la démo :** les mêmes chiffres sur vos propres verres mesurés
au pied à coulisse, via `python3 scripts/eval_mm.py --real pictures` (voir
« Évaluer » plus bas). Les chiffres ci-dessus viennent de captures rendues, dont
la vérité terrain est exacte par construction ; ils valident la géométrie, pas le
papier ni l'appareil photo.

---

## Comment ça marche

```
photo
  ├─ 4 marqueurs ArUco  →  homographie (DLT, 16 correspondances)  →  échelle en mm
  ├─ zone de pose redressée à 6 px/mm
  ├─ modèle U-Net (TFJS)  →  « où est le verre »      ← robustesse
  ├─ réponse de bord + tracé radial (DP)  →  « où exactement est son bord »  ← précision
  └─ contour en mm  →  A, B, périmètre, SVG 1:1, monture STL
```

### Le choix qui structure tout : pas d'OpenCV.js

La première version chargeait les 10,2 Mo d'OpenCV.js et faisait tourner
détection, redressement, Canny et remplissage **sur le thread principal**. Un
téléphone milieu de gamme se figeait. Tout a été réécrit en TypeScript :
homographie (DLT par équations normales, normalisation de Hartley), contours
sous-pixel (marching squares), rééchantillonnage bilinéaire. Le chargement
initial est passé de 10,2 Mo à **environ 420 Ko** (130 Ko compressés), et tout le
calcul tourne dans un **Web Worker** — l'interface reste utilisable pendant la
mesure, ce que le jury peut vérifier en faisant défiler la page.

### On ne redresse plus l'image pour mesurer

On trouve les marqueurs, on en déduit l'homographie, et on ne redresse que la
zone de pose. Les ~200 points du contour sont ensuite exprimés en millimètres
sans rééchantillonner la photo entière.

### Pourquoi la feuille est grise et unie

L'idée naturelle est d'imprimer un motif fin et de chercher le verre à la
réfraction de ce motif. **La physique ne suit pas** : posé sur le papier, un
verre voit son fond à une distance quasi nulle, et la déviation prismatique est
proportionnelle à cette distance — un verre de 4 dioptries à 2 mm du papier
décale le motif d'environ **20 micromètres**. Nous avons construit, testé et
abandonné cette approche. Un motif n'apporte donc aucun signal et ajoute du
désordre là où le seul indice vraiment fort se trouve : **le bord du verre**, un
biseau de verre courbe qui assombrit le fond et renvoie une caustique. Gris
moyen, pour que l'assombrissement et la caustique aient tous deux de la place.

### Le bord mesure, le modèle oriente

- `segment.ts` isole le relief du bord (passe-bande à l'échelle du biseau,
  normalisé par l'éclairement local — donc insensible à une lampe sur le côté).
- `radialRidge.ts` cherche la **crête fermée** la plus marquée autour du verre,
  par programmation dynamique en polaire. Le résultat est fermé et lisse par
  construction : une portion de bord effacée par un reflet est simplement
  traversée. Et la crête d'un biseau est son milieu, donc le contour ne dépend
  pas de la largeur du biseau — qui varie d'un facteur trois entre deux verres.
- Le **modèle** (Palier 2) fournit le point de départ du tracé. Il entre à
  256 × 256 pour 156 × 176 mm, soit 0,6 mm par pixel : il ne peut pas mesurer,
  et ce n'est pas son rôle. Il repère, le bord mesure. Si le bord est
  inexploitable, l'app retombe sur le contour du modèle et l'affiche comme tel.

### N'importe quelle imprimante

La feuille n'a pas besoin d'être imprimée à 100 %. La plupart des pilotes
réduisent la page sans le dire — la nôtre est sortie à 84 % — et tout ce qui est
imprimé est réduit du même facteur, donc toute mesure le serait aussi,
silencieusement. Mesurer le trait de contrôle au pied à coulisse et saisir sa
vraie longueur suffit : l'app applique le facteur à l'échelle du plan redressé
et tout le reste suit, sans qu'aucun autre calcul ait à savoir qu'une correction
a eu lieu. La valeur est retenue d'une session à l'autre.

### L'app refuse plutôt que de deviner

Un instrument qui dit parfois « reprenez la photo » est utile ; un instrument qui
annonce parfois 63 mm pour un verre de 49 mm sans le moindre signe ne l'est pas.
Trois garde-fous indépendants, parce qu'aucun ne suffit seul :

1. **Score de suivi du bord** — à quel point le contour longe vraiment une crête.
2. **Crête concurrente à l'extérieur** — un verre montre souvent un anneau
   intérieur (sa caustique, un reflet annulaire) plus brillant que son vrai bord.
   Le score seul en est content : il a bien suivi une crête fermée. Cette mesure
   pose la question complémentaire — reste-t-il quelque chose de bordé plus
   loin ? Au-delà d'un vrai bord il n'y a que du papier.
3. **Désaccord avec le modèle** (actif une fois le modèle entraîné) — le modèle
   n'a aucun intérêt dans telle ou telle crête. Quand lui et le tracé divergent
   d'un quart, l'un des deux a tort et aucun chiffre n'est annoncé.

Plus : aire plausible, compacité (un verre a un rapport isopérimétrique proche de
0,9), et résidu de l'ajustement d'échelle affiché à l'écran.

---

## Dossier données et IA

| Élément | Détail |
|---|---|
| Jeu de données | 100 % produit par nous : rendus synthétiques (`scripts/synthetic.py`) + photos de nos propres verres étiquetées à la main (`scripts/label_lens.py`) |
| Licence des données | ce dépôt ; **aucune donnée personnelle** (ni visage, ni nom, ni ordonnance) |
| Modèle pré-entraîné | aucun — rien à citer, le U-Net est entraîné de zéro |
| Architecture | U-Net ~150 000 paramètres, entrée 256 × 256 en niveaux de gris |
| Entraînement | `notebook/train_unet.ipynb`, Google Colab, perte Dice + BCE |
| Exécution | TensorFlow.js, backend **WebAssembly**, dans le Web Worker |
| Bibliothèques | js-aruco2 (MIT, vendorisée dans `src/lib/vendor/jsAruco2/`), manifold-3d, three.js, TensorFlow.js |

Le rendu synthétique modélise ce qui rend un verre visible — biseau sombre et
caustique au bord, perte de contraste de Fresnel d'environ 8 % à la traversée,
reflets spéculaires, ombre portée, réfraction quasi nulle — et varie éclairage,
exposition, netteté, point de vue et compression JPEG. C'est ce qui permet
d'entraîner sur des milliers de cas là où une journée de photos en donnerait
quelques dizaines.

**À compléter après entraînement :** IoU de validation, erreur du modèle seul,
taille du modèle, temps d'inférence sur un vrai téléphone. Si le modèle fait
moins bien que le chemin classique, le dire et livrer le chemin classique.

---

## Lancer et vérifier

```bash
npm install
npm run dev            # développement
npm run build          # production (dist/)
npm run lint
npm run selftest       # monture : maillage fermé + STL valide ; export SVG 1:1
npm run harness        # recompile le harnais d'évaluation après un changement .ts
npm run sheet          # régénère la feuille de capture

# Déploiement sous un sous-chemin (GitHub Pages projet) :
OPTIFRAME_BASE=/optiframe/ npm run build
```

Pour tester la caméra sur un vrai téléphone, servez en HTTPS (tunnel Cloudflare
ou ngrok) : les navigateurs bloquent `getUserMedia` en HTTP.

### Régénérer la feuille de capture

```bash
pip install -r scripts/requirements.txt
python3 scripts/make_marker_sheet.py
```

Écrit `public/capture-sheet.pdf`, son aperçu PNG, et `src/lib/captureSheet.json`
— la géométrie en millimètres que l'app lit. Feuille et app ne peuvent donc pas
diverger.

### Évaluer en millimètres

```bash
node scripts/headless/build.mjs        # après toute modification d'un .ts

python3 scripts/eval_mm.py --synthetic 40            # géométrie, vérité exacte
python3 scripts/eval_mm.py --real pictures --sweep   # vos verres, pied à coulisse
```

Pour `--real`, placez `pictures/truth.json` à côté des photos :

```json
{ "IMG_0001.jpg": { "aMm": 49.8, "bMm": 34.2 } }
```

Le harnais exécute **le code TypeScript compilé que le navigateur exécute** — pas
une réimplémentation Python. Un seuil réglé ici donne le même chiffre dans l'app.

### Entraîner le modèle

Marche à suivre complète : **[ENTRAINER-LE-MODELE.md](ENTRAINER-LE-MODELE.md)**.
En résumé :

```bash
npm run harness
python3 scripts/make_dataset.py --synthetic 3000 --out data/train
python3 scripts/label_lens.py --photos pictures/aruco --out data/real
# puis notebook/train_unet.ipynb sur Colab, et décompresser le résultat
# dans public/model/
```

---

## Limites connues

- Les chiffres de précision ci-dessus viennent de captures **rendues**. La
  vérité terrain y est exacte, mais elles ne valident ni le papier, ni l'optique
  réelle de l'appareil photo. À refaire au pied à coulisse avant la démo.
- Environ une capture rendue sur sept est **refusée** plutôt que mesurée (bord
  trop faible, forme implausible). Sur les huit photos réelles, aucune.
- Environ **1 capture rendue sur 40 donne un chiffre faux de plusieurs
  millimètres en passant les deux garde-fous classiques** — le tracé suit une
  crête convaincante autour de la mauvaise chose. C'est précisément ce que le
  troisième garde-fou (désaccord avec le modèle) existe pour attraper, et il
  reste inactif tant que le modèle n'est pas entraîné. Aucune occurrence sur les
  huit photos réelles. C'est voulu, mais cela veut dire qu'il faut parfois
  reprendre la photo.
- Un verre teinté ou fortement traité n'a pas été testé : son bord se comporte
  autrement. C'est précisément le cas que le modèle doit rattraper.
- L'alignement vertical des deux verres suit la convention « boxing » (centres
  des boîtes alignés). Aucun axe pupillaire n'est mesuré.
- La monture ne modélise pas de vraie rainure de clipsage, seulement un jeu
  uniforme de 0,2 mm.
- Le repli sans imprimante (quatre coins à la main sur une carte bancaire) est
  nettement moins précis ; il existe pour ne pas rester bloqué, pas pour mesurer.

---

## Checklist de remise

- [ ] URL HTTPS qui s'ouvre sur un téléphone neuf, sans installation
- [ ] QR code prêt (onglet À propos)
- [x] Dispositif de capture remontable en 2 min (une feuille A4 à imprimer)
- [x] `monture.stl` téléchargeable depuis l'app
- [x] Export du contour en SVG 1:1
- [ ] Chiffres de précision sur vos propres verres, au pied à coulisse
- [ ] Modèle entraîné déposé dans `public/model/`
- [ ] Démo répétée, chaque membre sait expliquer sa partie
