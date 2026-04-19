# Beer Peer

Deep Learning Project - RSU

*Photograph your food. Discover your perfect beer.*

---

## 1. Project Idea

Beer Peer is a deep learning application that bridges the gap between what a person is eating and what they should be drinking. The core concept is simple: a user photographs their meal, and the project returns a beer style recommendation grounded in real community tasting language.

The motivation comes from a genuine everyday moment. You sit down at a restaurant, or you are cooking at home, and you wonder what to drink. Wine pairing is well documented. Beer pairing is equally valid but far less accessible to the average person. Beer Peer makes that knowledge available through a photograph.

The project is academically grounded in two independent machine learning tasks - image classification and text classification - connected by a domain-specific business logic layer. This architecture satisfies the dual-dataset and CNN+LSTM requirement of the project rubric while producing a product concept with genuine real-world utility.

---

## 2. Product

### 2.1 User Flow

The interaction is designed to take under 10 seconds from input to output:

1. User opens Beer Peer and photographs their meal on a plate or table.
2. The CNN model identifies the dish from the image.
3. A lookup table maps the dish to two or three compatible beer styles based on established flavor pairing principles.
4. The LSTM model surfaces representative language from BeerAdvocate reviewers describing each recommended style.
5. The user sees the dish name, beer style recommendations, and a short flavor description for each style in the words of real beer drinkers.

### 2.2 Example Output

| | |
| --- | --- |
| **Input** | Photo of pizza margherita |
| **CNN Output** | Pizza |
| **Pairing Logic** | Pizza pairs with: Lager, IPA, Amber Ale |
| **LSTM - Lager** | Crisp, clean, light bitterness. Reviewers note it lifts salty and cheesy flavors without overpowering the dish. |
| **LSTM - IPA** | Citrusy, hoppy, with a bitter finish that cuts through rich tomato sauce and melted cheese. |

---

## 3. Datasets

### 3.1 Image Dataset - Food-101

| | |
| --- | --- |
| **Name** | Food-101 |
| **Source** | ETH Zurich / Hugging Face Hub |
| **Load command** | `load_dataset("ethz/food101")` |
| **Size** | 101,000 images across 101 food categories |
| **Classes** | 101 dishes: pizza, sushi, steak, burger, ramen, curry, salad, pasta and more |
| **Split** | 750 training images and 250 test images per class |
| **Format** | RGB images, variable resolution, pre-labeled by class folder |

Food-101 is one of the most widely used food image classification benchmarks in computer vision research. Its class labels map directly and naturally to flavor profiles, enabling a clean handoff to the pairing logic layer. The dataset is accessed in a single line of code with no preprocessing required for initial loading.

### 3.2 Text Dataset - BeerAdvocate Reviews

| | |
| --- | --- |
| **Name** | Beer Reviews - BeerAdvocate |
| **Source** | Kaggle |
| **Load command** | `kaggle datasets download rdoume/beerreviews` |
| **Size** | 1.5 million reviews spanning more than 10 years |
| **Key columns** | beer_style, review_aroma, review_taste, review_palate, review_overall, review_text |
| **Beer styles** | 100+ styles including IPA, Stout, Lager, Pilsner, Wheat, Sour, Porter, Amber, Hefeweizen |
| **License** | Publicly available via Kaggle |

BeerAdvocate is one of the largest community beer review datasets available. Each review includes structured numeric ratings across five sensory dimensions as well as free-form text. The `beer_style` column serves as the LSTM classification label. For the project, styles will be grouped into 8-10 meaningful macro-categories to reduce sparsity and improve model performance.

---

## 4. High-Level Architecture

The project pipeline consists of three sequential layers: a visual classification layer, a business logic layer, and a textual characterization layer. Each layer is independently trained and evaluated before being connected into the full pipeline.

| Step | Component | Function | Dataset |
| --- | --- | --- | --- |
| 1 | CNN - Image Classifier | Takes a food photograph as input. Outputs a food category label from 101 classes. | Food-101 |
| 2 | Business Logic Layer | Lookup table mapping each food class to 2-3 compatible beer style categories based on flavor pairing principles. | Hardcoded mapping |
| 3 | LSTM - Text Classifier | Trained on BeerAdvocate reviews to learn style-specific flavor language. Outputs a representative tasting profile per beer style. | BeerAdvocate |
| 4 | Output Layer | Combines CNN label + lookup result + LSTM tasting language into a structured user-facing recommendation. | Combined |

### 4.1 CNN Architecture

The CNN uses a transfer learning approach based on a pre-trained backbone such as ResNet-50 or EfficientNet-B0, with the classification head fine-tuned on Food-101. Transfer learning is chosen because the base models carry strong general visual features from ImageNet training, significantly reducing the training time and data required to reach competitive accuracy on food classification.

- Input: RGB food image, resized to 224x224
- Backbone: ResNet-50 pre-trained on ImageNet (frozen in early epochs)
- Classification head: Dense layer with 101 output units, softmax activation
- Output: Food category label with confidence score

### 4.2 LSTM Architecture

The LSTM is trained on BeerAdvocate review text to classify text into beer style categories. The model learns to associate descriptive language - hoppy, bitter, citrusy, roasty, malty, crisp - with style labels. At inference time, rather than classifying new text, the model is used to retrieve representative review sentences per style, presenting community-sourced language to the user.

- Input: Tokenized review text from BeerAdvocate, padded to fixed length
- Embedding: Trainable word embedding layer (dimension 128)
- Recurrent layer: Bidirectional LSTM (128 units)
- Output: Beer style category from 8-10 macro-style classes
- Inference mode: Style-conditioned text retrieval from review corpus

### 4.3 Business Logic Layer

The lookup table is a small, manually curated mapping from Food-101 class names to beer style categories. It is grounded in established craft beer pairing principles from the Brewers Association and industry sources.

| Food | Recommended Beer Styles |
| --- | --- |
| Pizza, burger, fried chicken | Lager, IPA, Amber Ale |
| Steak, BBQ ribs, grilled meat | Stout, Porter, Brown Ale |
| Sushi, seafood, oysters | Pilsner, Hefeweizen, Wheat Beer |
| Spicy curry, hot wings | IPA, Pale Ale |
| Caesar salad, grilled vegetables | Saison, Witbier |
| Chocolate desserts, brownies | Imperial Stout, Porter |

---

## 5. Technical Stack

| | |
| --- | --- |
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch or TensorFlow/Keras |
| **Image Processing** | torchvision, PIL, Albumentations |
| **NLP** | NLTK, spaCy for preprocessing; custom LSTM in PyTorch/Keras |
| **Data Loading** | Hugging Face datasets library, Kaggle API |
| **Experiment Tracking** | Matplotlib, Seaborn for loss and accuracy visualization |

---

## 6. Success Criteria

- CNN achieves above 70% top-1 accuracy on Food-101 test set (state of the art with transfer learning reaches 90%+, 70% is the baseline target for this project scope).
- LSTM achieves above 75% accuracy in classifying BeerAdvocate review text by beer style macro-category.
- The end-to-end pipeline produces a plausible beer style recommendation for at least 80 of the 101 Food-101 classes covered by the lookup table.
- The system can process a single image input and return a full recommendation output in under 5 seconds on a standard laptop CPU.

---

Beer Peer - Project Description v1.0 - RSU Deep Learning Course
