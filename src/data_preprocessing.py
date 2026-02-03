class ODIRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Adjust if your CSV uses N/D/G/... instead
        self.label_cols = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract',
                           'AMD', 'Hypertension', 'Myopia', 'Others']

        self.samples = []

        for _, row in self.df.iterrows():
            img_id = str(row['ID'])

            label = row[self.label_cols].values.astype(float)

            left_img = os.path.join(self.img_dir, f"{img_id}_left.jpg")
            right_img = os.path.join(self.img_dir, f"{img_id}_right.jpg")

            if os.path.exists(left_img):
                self.samples.append((left_img, label, 'left'))

            if os.path.exists(right_img):
                self.samples.append((right_img, label, 'right'))

        print(f"Loaded {len(self.samples)} valid image-label pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, side = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label
