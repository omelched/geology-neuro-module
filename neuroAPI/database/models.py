import uuid
import enum
from datetime import datetime

from sqlalchemy import Column, String, Numeric, ForeignKey, Integer, CheckConstraint, UniqueConstraint, Text, \
    LargeBinary, Index, DateTime, select
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.types import CHAR, Enum
from sqlalchemy_utils import force_instant_defaults

from neuroAPI.database.ext import GUID
from neuroAPI.database import database_handler

Base = declarative_base()
force_instant_defaults()


# ENUMS

class UserStatus(enum.Enum):
    admin = 1
    basic = 2


class ContactInformationType(enum.Enum):
    firstname = 1
    lastname = 2
    email = 3


class BorderPointType(enum.Enum):
    max = 1
    min = 2


class MetricType(enum.Enum):
    class_stat = 1
    overall_stat = 2


# TABLES
# TODO: add back_populates

class AppUser(Base):
    __tablename__ = 'app_users'
    __table_args__ = {
        'comment':
            ("Stores user data. "
             "Not named 'users' because of PostgreSQL keyword.")
    }

    id = Column(GUID,
                primary_key=True, default=uuid.uuid4,
                comment='User id')
    username = Column(String(64),
                      nullable=False, unique=True,
                      comment='User name')
    salt = Column(CHAR(16),
                  nullable=False,
                  comment='Password salt')
    password = Column(CHAR(64),
                      nullable=False,
                      comment='sha-256("salt"+":"+"user password")')
    user_status = Column(Enum(UserStatus),
                         nullable=False,
                         comment='Sets privileges')


class UserContactInformation(Base):
    __tablename__ = 'user_contact_information'
    __table_args__ = {
        'comment': "Stores user contact information."
    }

    user_id = Column(ForeignKey('app_users.id'),
                     primary_key=True,
                     comment='User id')
    contact_info_type = Column(Enum(ContactInformationType),
                               primary_key=True,
                               comment='CI type')
    contact_info_value = Column(String(320),
                                nullable=True,
                                comment='CI value')


class Deposit(Base):
    __tablename__ = 'deposits'
    __table_args__ = {
        'comment': "Stores deposit data."
    }

    id = Column(GUID,
                primary_key=True, default=uuid.uuid4,
                comment='Deposit id')
    username = Column(String(64),
                      nullable=False,
                      comment='Deposit name')


class DepositOwners(Base):
    __tablename__ = 'deposit_owners'
    __table_args__ = {
        'comment': "Links users and owned deposits."
    }

    deposit_id = Column(ForeignKey('deposits.id'),
                        primary_key=True,
                        comment='Deposit id')

    user_id = Column(ForeignKey('app_users.id'),
                     primary_key=True,
                     comment='User id')


class DepositBorders(Base):
    __tablename__ = 'deposit_borders'
    __table_args__ = {
        'comment': "Stores deposit borders data."
    }

    deposit_id = Column(ForeignKey('deposits.id'),
                        primary_key=True,
                        comment='Deposit id')
    point_type = Column(Enum(BorderPointType),
                        primary_key=True,
                        comment='Border point type')
    x_value = Column(Numeric,
                     nullable=False,
                     comment='Point value on x-axis')
    y_value = Column(Numeric,
                     nullable=False,
                     comment='Point value on y-axis')
    z_value = Column(Numeric,
                     nullable=False,
                     comment='Point value on z-axis')


class DepositFiles(Base):
    __tablename__ = 'deposit_files'
    __table_args__ = {
        'comment': "Lists links to deposit’s files."
    }

    deposit_id = Column(ForeignKey('deposits.id'),
                        primary_key=True,
                        comment='Deposit id')
    file_id = Column(ForeignKey('files.id'),
                     primary_key=True,
                     comment='File id')


class Rock(Base):
    __tablename__ = 'rocks'
    __table_args__ = {
        'comment': "Store rock data."
    }

    id = Column(GUID,
                primary_key=True, default=uuid.uuid4,
                comment='Rock id')
    deposit_id = Column(ForeignKey('deposits.id'),
                        nullable=False,
                        comment='Deposit id')
    index = Column(Integer,
                   nullable=False, autoincrement=True,
                   comment='Rock index in deposit')
    name = Column(String(64),
                  nullable=False,
                  comment='Rock name')
    color = Column(CHAR(7), CheckConstraint('color IN NULL or LIKE #%'),
                   nullable=True,
                   comment='Rock hex color, e.g. "#FFFFFF"')


class Well(Base):
    __tablename__ = 'wells'
    __table_args__ = {
        'comment': "Store wells."
    }

    id = Column(GUID,
                primary_key=True, default=uuid.uuid4,
                comment='Well id')
    deposit_id = Column(ForeignKey('deposits.id'),
                        nullable=False,
                        comment='Deposit id')
    head_x = Column(Numeric,
                    nullable=False,
                    comment='Head point value on x-axis')
    head_y = Column(Numeric,
                    nullable=False,
                    comment='Head point value on y-axis')
    head_z = Column(Numeric,
                    nullable=False,
                    comment='Head point value on z-axis')
    tail_x = Column(Numeric,
                    nullable=False,
                    comment='Tail point value on x-axis')
    tail_y = Column(Numeric,
                    nullable=False,
                    comment='Tail point value on y-axis')
    tail_z = Column(Numeric,
                    nullable=False,
                    comment='Tail point value on z-axis')


class WellIntervals(Base):
    __tablename__ = 'well_intervals'
    __table_args__ = {
        'comment': "Lists well’s intervals."
    }

    well_id = Column(ForeignKey('wells.id'),
                     primary_key=True,
                     comment='Well id')
    position = Column(Integer,
                      primary_key=True,
                      comment='Interval position from head')
    from_x = Column(Numeric,
                    nullable=False,
                    comment='From point value on x-axis')
    from_y = Column(Numeric,
                    nullable=False,
                    comment='From point value on y-axis')
    from_z = Column(Numeric,
                    nullable=False,
                    comment='From point value on z-axis')
    to_x = Column(Numeric,
                  nullable=False,
                  comment='To point value on x-axis')
    to_y = Column(Numeric,
                  nullable=False,
                  comment='To point value on y-axis')
    to_z = Column(Numeric,
                  nullable=False,
                  comment='To point value on z-axis')


class KnownBlock(Base):
    __tablename__ = 'known_blocks'
    __table_args__ = (
        UniqueConstraint('well_id', 'size', 'center_x', 'center_y', 'center_z'),
        {
            'comment': "Stores known blocks."
        }
    )

    id = Column(GUID,
                primary_key=True, default=uuid.uuid4,
                comment='Known block id')
    well_id = Column(ForeignKey('wells.id'),
                     nullable=False,
                     comment='This block well id')
    size = Column(Numeric,
                  nullable=False,
                  comment='Block size')
    center_x = Column(Numeric,
                      nullable=False,
                      comment='Center point value on x-axis')
    center_y = Column(Numeric,
                      nullable=False,
                      comment='Center point value on y-axis')
    center_z = Column(Numeric,
                      nullable=False,
                      comment='Center point value on z-axis')
    content = Column(ForeignKey('rocks.id'),
                     nullable=False,
                     comment='Rock on this block')


class CrossValidation(Base):
    __tablename__ = 'cross_validations'
    __table_args__ = {
        'comment': "Stores cross-validations."
    }

    id = Column(GUID,
                primary_key=True, default=uuid.uuid4,
                comment='Cross-validation id')
    name = Column(String(64),
                  nullable=False,
                  comment='Cross-validation name')


class Metric(Base):
    __tablename__ = 'metrics'
    __table_args__ = {
        'comment': "Stores metrics."
    }

    id = Column(GUID,
                primary_key=True, default=uuid.uuid4,
                comment='Metric id')
    name = Column(String(64),
                  nullable=False,
                  unique=True,
                  comment='Metric name')
    description = Column(Text,
                         nullable=True,
                         comment='Metric description, e.g. formulae')
    mtype = Column(Enum(MetricType), name='type',
                  nullable=False,
                  comment='Metric type')


class NeuralModel(Base):
    __tablename__ = 'neural_models'
    __table_args__ = {
        'comment': "Stores neural models."
    }

    id = Column(GUID,
                primary_key=True, default=uuid.uuid4,
                comment='Neural model id')
    deposit_id = Column(ForeignKey('deposits.id'),
                        nullable=False,
                        comment='Deposit id')
    block_size = Column(Numeric,
                        nullable=False,
                        comment='Neural model block size')
    max_epochs = Column(Integer,
                        nullable=False,
                        comment='Max epoch count')
    cross_validation_id = Column(ForeignKey('cross_validations.id'),
                                 nullable=True,
                                 comment='Cross-validation grouping entity id')
    structure = Column(LargeBinary,
                       nullable=False,
                       comment='NM structure')
    weights = Column(LargeBinary,
                     nullable=False,
                     comment='NM weights')


class NeuralModelExcludedWells(Base):
    __tablename__ = 'neural_models_excluded_wells'
    __table_args__ = {
        'comment': "Lists excluded wells from training."
    }

    neural_model_id = Column(ForeignKey('neural_models.id'),
                             primary_key=True,
                             comment='Neural model id')
    well_id = Column(ForeignKey('wells.id'),
                     primary_key=True,
                     comment='Well id')


class NeuralModelMetrics(Base):
    __tablename__ = 'neural_models_metrics'
    __table_args__ = {
        'comment': "Lists metric data."
    }

    neural_model_id = Column(ForeignKey('neural_models.id'),
                             primary_key=True,
                             comment='Neural model id')
    metric_id = Column(ForeignKey('metrics.id'),
                       primary_key=True,
                       comment='Metric id')
    epoch = Column(Integer,
                   primary_key=True,
                   comment='Current epoch')
    rock_id = Column(ForeignKey('rocks.id'),
                     nullable=True,
                     comment='Rock id (if metric.type = class_stat))')
    value = Column(Text,
                   nullable=False,
                   comment='Metric value')

    @staticmethod
    def get_create_metric(name: str, mtype: MetricType, session: Session = None) -> uuid.UUID:
        standalone = False

        if not session:
            session = database_handler.get_session()
            standalone = True

        result = session.execute(select(Metric).where(Metric.name == name))

        if result:
            idx = result.fetchone()[0].id
            if standalone:
                session.rollback()
            return idx

        metric = Metric(name=name, mtype=mtype)

        try:
            session.add(metric)
        except Exception as e:  # TODO: custom exeptions with sys.exc_info()[0]
            session.rollback()
            raise e
        else:
            if standalone:
                session.commit()

        return metric.id


class PredictedBlock(Base):
    __tablename__ = 'predicted_blocks'
    __table_args__ = (
        UniqueConstraint('neural_model_id', 'center_x', 'center_y', 'center_z'),
        Index('known_block_index', 'known_block_id', postgresql_using='hash'),
        {
            'comment': "Stores predicted blocks."
        }
    )

    id = Column(GUID,
                primary_key=True, default=uuid.uuid4,
                comment='Predicted block id')
    neural_model_id = Column(ForeignKey('neural_models.id'),
                             comment='Neural model id')
    center_x = Column(Numeric,
                      nullable=False,
                      comment='Center point value on x-axis')
    center_y = Column(Numeric,
                      nullable=False,
                      comment='Center point value on y-axis')
    center_z = Column(Numeric,
                      nullable=False,
                      comment='Center point value on z-axis')
    known_block_id = Column(ForeignKey('known_blocks.id'),
                            nullable=True,
                            comment='paired Known block')
    content = Column(ForeignKey('rocks.id'),
                     nullable=False,
                     comment='Rock on this block')


class PredictedBlocksOutputs(Base):
    __tablename__ = 'predicted_blocks_outputs'
    __table_args__ = {
        'comment': "Lists predicted block outputs."
    }

    predicted_block_id = Column(ForeignKey('predicted_blocks.id'),
                                primary_key=True,
                                comment='Predicted block id')
    rock_id = Column(ForeignKey('rocks.id'),
                     primary_key=True,
                     comment='Rock id')
    value = Column(Numeric,
                   nullable=False,
                   comment='probability [0, 1]')


class ContentType(Base):
    __tablename__ = 'content_types'
    __table_args__ = {
        'comment': ("Stores MIME content_types, e.g.:"
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet  //.xlsx"
                    "application/vnd.ms-excel  //.xls"
                    "application/vnd.ms-excel.sheet.binary.macroEnabled.12 //.xlsb"
                    "text/csv  //.csv")
    }

    id = Column(GUID,
                primary_key=True, default=uuid.uuid4,
                comment='Content type id')
    name = Column(String(127),
                  nullable=False, unique=True,
                  comment='MIME content type')


class File(Base):
    __tablename__ = 'files'
    __table_args__ = (
        UniqueConstraint('name', 'data_type'),
        Index('file_index', 'name', 'data_type'),
        {
            'comment': "Stores files."
        }
    )

    id = Column(GUID,
                primary_key=True, default=uuid.uuid4,
                comment='File id')
    name = Column(String(255),
                  nullable=False,
                  comment='original filename with extension, e.g. "text.xlsx"')
    data_type = Column(ForeignKey('content_types.id'),
                       nullable=False,
                       comment='MIME content type')
    description = Column(Text,
                         nullable=True,
                         comment='Long description')
    date_added = Column(DateTime,
                        nullable=False, default=datetime.now(),
                        comment='When was created')
    content = Column(LargeBinary,
                     nullable=False,
                     comment='File itself in binary')
