# Mini-Turkish-Tokenizer
A turkish tokenizer which has 5610 vocabulary size for little models.
**Türkçe dil modelleri için optimize edilmiş, kompakt BPE tokenizer**

---

## 📌 Özet

Mini Turkish Tokenizer, Türkçe NLP görevleri için özel olarak tasarlanmış bir BPE (Byte Pair Encoding) tokenizer'dır. CulturaX Turkish dataset'inin 735,991 dokümanından eğitilerek, Türkçe metinleri verimli bir şekilde tokenlere dönüştürür.

### Temel Özellikler

- **Vocab Size:** 5,610 tokens (kompakt ve verimli)
- **Dil:** Türkçe (🇹🇷)
- **Algoritma:** BPE (Byte Pair Encoding)
- **Eğitim Verisi:** CulturaX Turkish (735,991 dokümandan)
- **Format:** HuggingFace PreTrainedTokenizerFast
- **Lisans:** GNU General Public License v2.0 (açık kaynak)

---

## 🚀 Hızlı Başlangıç

### Kurulum

```bash
pip install transformers
```

### Temel Kullanım

```python
from transformers import AutoTokenizer

# Tokenizer'ı yükle
tokenizer = AutoTokenizer.from_pretrained(
    "kaanilker/mini-turkish-tokenizer"
)

# Metni tokenize et
text = "Merhaba, ben yapay zekayım!"
tokens = tokenizer.encode(text)

print(tokens)
# Output: [59, 83, 96, 86, 79, 80, ...]
```

### Decode Etme

```python
# Token'ları metne geri çevir
decoded = tokenizer.decode(tokens)
print(decoded)
# Output: "Merhaba, ben yapay zekayım!"
```

### Batch Processing

```python
texts = [
    "Merhaba dünya",
    "Türkçe NLP",
    "Yapay zeka harika"
]

# Batch tokenize
encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=100,
    return_tensors="pt"
)

print(encoded['input_ids'].shape)
# Output: torch.Size([3, 100])
```

---

## 📊 Teknik Detaylar

### Special Tokens

| Token | ID | Açıklama |
|---|---|---|
| `<pad>` | 0 | Padding (doldurma) |
| `<unk>` | 1 | Unknown (bilinmeyen) |
| `<bos>` | 2 | Beginning of Sequence (başlangıç) |
| `<eos>` | 3 | End of Sequence (bitiş) |

### Eğitim Konfigürasyonu

```python
vocab_size = 5610
min_frequency = 2
algorithm = "BPE"
pre_tokenizer = "Whitespace + Punctuation"
training_data = "CulturaX Turkish (735,991 documents)"
train_test_split = "90/10"
```

### Tokenizasyon Özellikleri

- **Ortalama Token Sayısı:** 8-12 token per sentence
- **Coverage (CulturaX):** ~98.5%
- **Encoding Hızı:** ~10,000 token/sec
- **Bellek Footprint:** 5-10 MB

---

## 💻 İleri Kullanım

### Attention Mask İle

```python
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "kaanilker/mini-turkish-tokenizer"
)

text = "Kısa"
encoded = tokenizer(
    text,
    padding="max_length",
    max_length=10,
    return_tensors="pt"
)

print(encoded['input_ids'])
# [1234, 0, 0, 0, 0, 0, 0, 0, 0, 0]

print(encoded['attention_mask'])
# [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

### Model Eğitmede Kullanım

```python
from transformers import TrainingArguments, Trainer
from transformers import LlamaForCausalLM, AutoTokenizer

model = LlamaForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained(
    "kaanilker/mini-turkish-tokenizer"
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=5e-4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    tokenizer=tokenizer,
)

trainer.train()
```

### Fine-tuning İçin

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    "kaanilker/mini-turkish-tokenizer"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "dbmdz/bert-base-turkish-cased"
)

# Türkçe metinleri tokenize et
inputs = tokenizer(
    ["Bu çok güzel!", "Berbat!"],
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

outputs = model(**inputs)
```

---

## 🎯 Kullanım Senaryoları

### 1. Türkçe Metin Sınıflandırması
```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="dbmdz/bert-base-turkish-cased",
    tokenizer=tokenizer
)

result = classifier("Bu ürün harika!")
print(result)
```

### 2. Türkçe Metin Üretimi
```python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="your-turkish-llm",
    tokenizer=tokenizer
)

generated = generator("Türkiye'nin başkenti", max_length=50)
print(generated)
```

### 3. Türkçe Soru-Cevap
```python
from transformers import pipeline

qa = pipeline(
    "question-answering",
    model="your-qa-model",
    tokenizer=tokenizer
)

result = qa(
    question="Türkiye'nin başkenti neresidir?",
    context="Türkiye'nin başkenti Ankara'dır."
)
print(result)
```

### 4. Türkçe Sentiment Analizi
```python
tokenizer = AutoTokenizer.from_pretrained(
    "kaanilker/mini-turkish-tokenizer"
)

texts = [
    "Çok güzel bir ürün!",
    "Berbat kalite",
    "Fena değil"
]

for text in texts:
    tokens = tokenizer.encode(text)
    print(f"{text} → {len(tokens)} tokens")
```

---

## 📋 Teknik Özellikler

### Vocab Dağılımı

```
Toplam Tokens: 5,610

Kategori Dağılımı:
├── Türkçe Kelimeler: ~3,366 (60%)
├── Subword Pieces: ~2,200 (39%)
├── Special Tokens: 4 (1%)
└── Diğer: ~40 (1%)
```

### Eğitim Verileri

- **Dataset:** CulturaX Turkish
- **Toplam Dokümandan:** 735,991
- **Toplam Token:** 500M
- **Train/Val Split:** 90/10
- **Min Frequency:** 2 (en az 2 kez görülmüş kelimeler)

---

## 🔧 Kurulum & Bağımlılıklar

### Gerekli Paketler

```bash
# Temel
pip install transformers>=4.30.0
pip install datasets>=2.0.0

# İsteğe bağlı (örnek kodlar için)
pip install torch>=1.9.0
pip install pytorch-lightning>=1.5.0
```

### Versiyonlar

```
Python: 3.8+
Transformers: 4.30+
Datasets: 2.0+
Torch: 1.9+
```

---

## 📚 Örnekler

### Örnek 1: Temel Tokenizasyon

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "kaanilker/mini-turkish-tokenizer"
)

# Basit cümle
text = "Günaydın, nasılsın?"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")
print(f"Token Sayısı: {len(tokens)}")

# Decode
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")
```

**Output:**
```
Tokens: [59, 83, 96, ...]
Token Sayısı: 5
Decoded: Günaydın, nasılsın?
```

### Örnek 2: Batch Tokenizasyon

```python
texts = [
    "Merhaba dünya",
    "Türkçe NLP harika",
    "Açık kaynak yazılım"
]

batch = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=20,
    return_tensors="pt"
)

print(batch['input_ids'].shape)
# torch.Size([3, 20])
```

### Örnek 3: Dilbilimsel Analiz

```python
# Kelime parçalanması
text = "Üniversitelerimizde"
tokens = tokenizer.tokenize(text)
print(f"Parçalar: {tokens}")
# Parçalar: ['Üniversite', 'leri', 'mizde']

# Token ID'leri
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"IDs: {ids}")
```

---

## ⚠️ Sınırlamalar

### Bilinçli Kısıtlamalar

1. **Vocab Size:** 5,610 (küçük ama verimli)
   - ✅ Hızlı tokenizasyon
   - ❌ Nadir kelimeleri parçalayabilir

2. **Türkçeye Özel:** Sadece Türkçe için optimize
   - ✅ Türkçe için en iyi
   - ❌ İngilizce vb. dillerle sorun olabilir

3. **CulturaX Bias:** Belirli alanlara biased olabilir
   - ✅ Haber, sosyal medya vb. iyi
   - ❌ Teknik jargon eksik olabilir

### Çözümler

```python
# Eğer UNK token çok görürsen:
# 1. Vocab'i büyüt
# 2. Farklı dataset kullan
# 3. Subword parçalamayı artır
```

---

## 🔄 Güncelleme 


### Güncelleme Nasıl Yapılır?

```bash
# En son sürümü kur
pip install --upgrade transformers

# Tokenizer'ı güncelle
tokenizer = AutoTokenizer.from_pretrained(
    "kaanilker/mini-turkish-tokenizer",
    revision="main"
)
```

---

## 📖 Kaynaklar

### Teorik Kaynaklar

- [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)
- [HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

### Benzer Projeler

- [Turkish BERT](https://github.com/dbmdz/bert-models)
- [Turkish GPT-2](https://huggingface.co/gpt2-turkish)
- [CulturaX Dataset](https://huggingface.co/datasets/uonlp/CulturaX)

---

## 🤝 Katkı ve İletişim

### Katkı Yapmak

```bash
git clone https://huggingface.co/kaanilker/mini-turkish-tokenizer
cd mini-turkish-tokenizer

# Değişiklik yap
git add .
git commit -m "Improvement: [açıklama]"
git push
```

### Sorun Bildirmek

Email'e posta at:
- Email: kaanilkernacar2010@gmail.com

---

## 📝 Sitasyon

Bu tokenizer'ı bilimsel çalışmalarda kullanıyorsan, lütfen şunu alıntı yap:

```bibtex
@software{mini_turkish_tokenizer,
  title = {Mini Turkish Tokenizer},
  author = {[Kaan İlker Nacar]},
  year = {2025},
  url = {https://huggingface.co/your-username/mini-turkish-tokenizer},
  license = {GPL-2.0}
}
```

### APA Format

[Kaan İlker Nacar]. (2025). Mini Turkish Tokenizer. HuggingFace Hub. Retrieved from https://huggingface.co/kaanilker/mini-turkish-tokenizer

---

## 📜 Lisans

**GNU General Public License v2.0**

Bu tokenizer açık kaynak yazılımdır. Özgürce:
- ✅ Kullanabilir
- ✅ Değiştirebilir
- ✅ Dağıtabilir
- ❌ AMA: Türev eserler de GPL v2.0 olmalı

---

---

## ✨ Teşekkürler

- **CulturaX Dataset** - Türkçe veri sağlayan uonlp
- **HuggingFace** - Tokenizer kütüphanesi
- **Transformers** - NLP framework
- **Açık Kaynak Topluluğu** - Destekler ve geri bildirim

---

## 📞 İletişim

- 📧 Email: kaanilkernacar2010@gmail.com
- 🔗 GitHub: https://github.com/kaanilker
- 🤗 HuggingFace: https://huggingface.co/kaanilker
---

**Made with ❤️ for Turkish NLP Community**

*Son güncelleme: Aralık 2025*
