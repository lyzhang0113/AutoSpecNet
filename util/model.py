import torch
import torch.nn as nn

class AutoSpecNet(nn.Module):
    def __init__(self, base, num_classes, num_years, num_makes, num_types):
        super().__init__()
        self.base = base

        if hasattr(base, 'fc'):
            in_features = self.base.fc.in_features
            self.base.fc = nn.Sequential()
        else:  # mobile net v2
            in_features = self.base.last_channel
            self.base.classifier = nn.Sequential()

        self.year_fc = nn.Sequential(
            nn.Dropout(p=0.2),
            # nn.ReLU(),
            nn.Linear(in_features, num_years)
        )

        self.make_fc = nn.Sequential(
            nn.Dropout(0.2),
            # nn.ReLU(),
            nn.Linear(in_features, num_makes)
        )

        self.type_fc = nn.Sequential(
            nn.Dropout(0.2),
            # nn.ReLU(),
            nn.Linear(in_features, num_types)
        )

        self.class_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features + num_years + num_makes + num_types, num_classes)
        )


    def forward(self, x):

        out = self.base(x)

        # fc = self.class_fc(out)

        year_fc = self.year_fc(out)
        make_fc = self.make_fc(out)
        type_fc = self.type_fc(out)

        fc = self.class_fc(torch.cat([out, year_fc, make_fc, type_fc], dim=1))

        return fc, year_fc, make_fc, type_fc