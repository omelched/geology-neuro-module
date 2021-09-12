# Generated by Django 3.2.7 on 2021-09-12 18:35

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone
import gnmproj.apps.gnm.src.fields


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('gnm', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='GnmUser',
            fields=[
                ('id', gnmproj.apps.gnm.src.fields.UUIDAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('jwt_issuance_time', models.DateTimeField(default=django.utils.timezone.now, verbose_name='JWT issuance time')),
                ('user', models.OneToOneField(editable=False, on_delete=django.db.models.deletion.CASCADE, related_name='gnm_user', to=settings.AUTH_USER_MODEL, verbose_name='user')),
            ],
            options={
                'verbose_name': 'GNM user',
                'verbose_name_plural': 'GNM users',
            },
        ),
    ]