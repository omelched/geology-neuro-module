import uuid
import enum

from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import declarative_base
from sqlalchemy.types import CHAR, Enum

from neuroAPI.database.ext import GUID

Base = declarative_base()


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


class MIMEDataType(enum.Enum):
    xlsx = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    xls = "application/vnd.ms-excel"
    xlsb = "application/vnd.ms-excel.sheet.binary.macroEnabled.12"
    csv = "text/csv"


# TABLES
# TODO: add back_populates

class AppUser(Base):
    __tablename__ = 'app_user'
    __table_args__ = {
        'comment':
            '''
            Stores user data.
            Not named "users" because of PostgreSQL keyword.
            '''
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
        'comment':
            '''
            Stores user contact information.
            '''
    }

    user_id = Column(ForeignKey('app_user.id'),
                     primary_key=True,
                     comment='User id')
    contact_info_type = Column(Enum(ContactInformationType),
                               primary_key=True,
                               comment='CI type')
    contact_info_value = Column(String(320),
                                nullable=True,
                                comment='CI value')

# TODO: other tables
