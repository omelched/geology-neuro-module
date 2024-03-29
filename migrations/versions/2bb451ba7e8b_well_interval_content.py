"""well_interval_content

Revision ID: 2bb451ba7e8b
Revises: b1f12a7c245b
Create Date: 2021-05-16 23:20:00.685828

"""
from alembic import op
import sqlalchemy as sa
import neuroAPI.database.ext


# revision identifiers, used by Alembic.
revision = '2bb451ba7e8b'
down_revision = 'b1f12a7c245b'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('well_intervals', sa.Column('content', neuroAPI.database.ext.GUID(), nullable=False,
                                              comment='Rock on this interval'))
    op.create_foreign_key(None, 'well_intervals', 'rocks', ['content'], ['id'])
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'well_intervals', type_='foreignkey')
    op.drop_column('well_intervals', 'content')
    # ### end Alembic commands ###
