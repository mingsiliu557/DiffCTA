<div align="center">



<h1>Leveraging Diffusion Models for Continual Test-Time Adaptation in Fundus Image Classification</h1>


</div>


<h2 style="text-align: left;">📌 Updates</h2>

**🗓 2025.03.19**: **Upload the code for Glaucoma&Diabetic classification.**  
**🗓 2025.03.06**: **Repository created.**  

## ✅ TODO  
- [ ] **Extend the experiment on segmentation task.**  
- [x] **Code will be released soon.**  ⏳



## 🛠️ Dependencies & Installation  

### 1️⃣ Clone the Repository  
```bash
git clone git@github.com:anonymous/DiffCTA.git
cd DiffCTA
```

### 2️⃣ Create Conda Environment & Install Dependencies  
```bash
conda create -n DiffCTA python=3.8 -y  
conda activate DiffCTA 
pip3 install -r requirements.txt  
```

## 🚀 Get Started  

### 📂 Dataset Preparation  

- Download the dataset using the following command:
```bash
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/data_lx/Fundus.zip
```
### Generate Adapted Image

```bash
bash optic_adapt.sh
```

### ⚡ Quick Test 🏂  
```

- Run the following command to perform a quick inference:  
```bash
bash TTA.sh
```

## 📊 Results  

| Method         | Domain A | Domain B | Domain C | Domain D | Domain E | AVG   |
|----------------|----------|----------|----------|----------|----------|--------|
| Source Only    | 68.37    | 50.68    | 65.74    | *34.98*  | *43.43*  | 52.64 |
| TENT           | 65.84    | *58.84*  | 60.36    | 28.42    | 42.76    | 51.24 |
| CoTTA          | 64.01    | 58.51    | 61.25    | 24.02    | 33.59    | 48.28 |
| EATA           | 66.46    | 58.50    | 63.34    | 33.41    | 40.42    | 52.43 |
| SAR            | 66.57    | 58.81    | 63.21    | 32.87    | 33.52    | 51.00 |
| DDA            | *69.71*  | 56.06    | *67.20*  | 34.22    | 39.73    | *53.38* |
| **DiffCTA**    | **70.47**| **59.34**| **68.46**| **35.59**| **45.93**| **55.96** |

## 📜 Citation (TODO)


## 📄 License  
The code and models are licensed under <a rel="license" href="./LICENSE">MIT License</a>. 

## 📬 Contact (anonymous)


## 🙌 Acknowledgement

The code is inspired by [VPTTA](https://github.com/Chen-Ziyang/VPTTA), [DDA](https://github.com/shiyegao/DDA)
