from mongoengine import Document, ReferenceField, EmbeddedDocumentField, EmbeddedDocument, ListField


class _BaseDocument(Document):
    meta = {
        'abstract': True,
    }


class _BaseSchema(EmbeddedDocument):
    meta = {
        'abstract': True,
    }


class Point(_BaseSchema):
    x = None
    y = None
    z = None


class Border(_BaseSchema):
    Name = None
    Min = None
    Max = None


class User(_BaseDocument):
    FirtsName = None


class Deposit(_BaseDocument):
    Offset = None
    Borders = ListField(EmbeddedDocumentField(Border))
    UsersID = ReferenceField(User)


class Rock(_BaseDocument):
    DepositID = ReferenceField(Deposit)
    Index = None
    Name = None
    Color = None
    UsersID = ReferenceField(User)


class Interval(_BaseSchema):
    rocksID = ReferenceField(Rock)
    azimuth = None
    zenith = None
    from_ = EmbeddedDocumentField(Point, db_field='from')
    to = EmbeddedDocumentField(Point)


class Well(_BaseDocument):
    DepositID = ReferenceField(Deposit)
    Name = None
    Head = EmbeddedDocumentField(Point)
    Intervals = ListField(EmbeddedDocumentField(Interval))
    Foot = EmbeddedDocumentField(Point)


class BlockModel(_BaseDocument):
    DepositID = ReferenceField(Deposit)
    Size = None


class FamousBlock(_BaseDocument):
    BlockModelID = ReferenceField(BlockModel)
    WellID = ReferenceField(Well)
    RockID = ReferenceField(Rock)
    Center = EmbeddedDocumentField(Point)


class NeuroModel(_BaseDocument):
    WellID = ReferenceField(Well)
    EpochsCount = None
    CrossValidationModel = None
    FullModels = None


class PredictedBlock(_BaseDocument):
    FamousBlockID = ReferenceField(FamousBlock)
    NeuroModelID = ReferenceField(NeuroModel)
    Outputs = None
    PredictedRockProbability = None


class Metricks(_BaseDocument):
    NeuroModelID = ReferenceField(NeuroModel)
    Epoch = None
    F1 = None
    Loss = None
    Accuraccy = None
