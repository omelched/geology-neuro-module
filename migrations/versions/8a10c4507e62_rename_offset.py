"""rename_offset

Revision ID: 8a10c4507e62
Revises: 0342fa05a31a
Create Date: 2021-05-16 22:41:15.991893

"""
from alembic import op
import sqlalchemy as sa
import neuroAPI.database.ext


# revision identifiers, used by Alembic.
revision = '8a10c4507e62'
down_revision = '0342fa05a31a'
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column('deposits', 'offset', new_column_name='bias')

    pass


def downgrade():
    op.alter_column('deposits', 'bias', new_column_name='offset')

    pass
