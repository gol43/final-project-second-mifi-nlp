import torch
import torch.nn as nn
import timm
from transformers import AutoModel
from torch.optim import AdamW


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.text_model = AutoModel.from_pretrained(
            config.TEXT_MODEL_NAME
        )

        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        # Обе модели выдают признаки разного размера, поэтому приводим их к одному размеру
        self.text_proj = nn.Linear(
            self.text_model.config.hidden_size,
            config.HIDDEN_DIM
        )

        self.image_proj = nn.Linear(
            self.image_model.num_features,
            config.HIDDEN_DIM
        )

        # В уроке здесь была классификация, а нам нужно получить одно число — калорийность
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )

    def forward(self, input_ids, attention_mask, image):
        text_features = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]

        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)

        # Тут тоже идём по примеру из урока и объединяем признаки обычным умножением
        fused_emb = text_emb * image_emb

        prediction = self.classifier(fused_emb)

        return prediction


def validate(model, val_loader, device):
    model.eval()

    val_loss = 0
    criterion = nn.L1Loss()

    # На validation только проверяем модель, поэтому градиенты нам здесь не нужны
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            predictions = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=images
            ).squeeze(1)

            loss = criterion(predictions, targets)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def train_model(model, train_loader, val_loader, config, device):
    # У нас регрессия, поэтому вместо кроссэнтропиии из классификации используем мае
    criterion = nn.L1Loss()

    # Как и в уроке, для разных частей модели оставляем свои lr
    optimizer = AdamW([
        {"params": model.text_model.parameters(), "lr": config.TEXT_LR},
        {"params": model.image_model.parameters(), "lr": config.IMAGE_LR},
        {"params": model.text_proj.parameters(), "lr": config.CLASSIFIER_LR},
        {"params": model.image_proj.parameters(), "lr": config.CLASSIFIER_LR},
        {"params": model.classifier.parameters(), "lr": config.CLASSIFIER_LR}
    ])

    model = model.to(device)

    best_val_loss = float("inf")

    for epoch in range(config.EPOCHS):
        model.train()

        train_loss = 0

        # Обычный цикл из уроков: prediction -> loss -> backward -> обновление весов
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()

            predictions = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image=images
            ).squeeze(1)

            loss = criterion(predictions, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # После каждой эпохи проверяем, как модель работает на validation
        val_loss = validate(
            model,
            val_loader,
            device
        )

        # Если результат стал лучше, сохраняем именно эту версию модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                config.SAVE_PATH
            )

        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} | "
            f"Train MAE: {train_loss:.2f} | "
            f"Val MAE: {val_loss:.2f}"
        )

    return model