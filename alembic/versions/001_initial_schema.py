"""Initial schema with all tables and user auth.

Revision ID: 001
Revises: None
Create Date: 2026-02-08

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.create_index("ix_user_email", "user", ["email"], unique=True)

    op.create_table(
        "sentencecache",
        sa.Column("sentence_hash", sa.String(), primary_key=True),
        sa.Column("sentence_text", sa.Text(), nullable=False),
        sa.Column("hiragana", sa.Text(), nullable=True),
        sa.Column("audio_path", sa.String(), nullable=True),
        sa.Column("hard_items_json", JSONB(), nullable=True),
        sa.Column("translations_json", JSONB(), nullable=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id"), nullable=True),
    )
    op.create_index("ix_sentencecache_user_id", "sentencecache", ["user_id"])

    op.create_table(
        "vocabentrycache",
        sa.Column("surface_hash", sa.String(), primary_key=True),
        sa.Column("surface", sa.String(), nullable=False),
        sa.Column("base_form", sa.String(), nullable=True),
        sa.Column("brief", sa.Text(), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("examples_json", JSONB(), nullable=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id"), nullable=True),
    )
    op.create_index("ix_vocabentrycache_user_id", "vocabentrycache", ["user_id"])

    op.create_table(
        "sourcematerial",
        sa.Column("source_hash", sa.String(), primary_key=True),
        sa.Column("title_ja", sa.String(), nullable=True),
        sa.Column("title_en", sa.String(), nullable=True),
        sa.Column("title_ko", sa.String(), nullable=True),
        sa.Column("title_zh", sa.String(), nullable=True),
        sa.Column("text_path", sa.String(), nullable=True),
        sa.Column("audio_path", sa.String(), nullable=True),
        sa.Column("category", sa.String(), nullable=True),
    )

    op.create_table(
        "usersettings",
        sa.Column("user_id", sa.String(), primary_key=True),
        sa.Column("default_voice", sa.String(), nullable=True),
        sa.Column("tts_speed", sa.Float(), nullable=True),
        sa.Column("show_fg", sa.Boolean(), nullable=True),
        sa.Column("show_hg", sa.Boolean(), nullable=True),
        sa.Column("default_source", sa.String(), nullable=True),
        sa.Column("font_scale", sa.Float(), nullable=True),
        sa.Column("native_lang", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("usersettings")
    op.drop_table("sourcematerial")
    op.drop_index("ix_vocabentrycache_user_id", table_name="vocabentrycache")
    op.drop_table("vocabentrycache")
    op.drop_index("ix_sentencecache_user_id", table_name="sentencecache")
    op.drop_table("sentencecache")
    op.drop_index("ix_user_email", table_name="user")
    op.drop_table("user")
