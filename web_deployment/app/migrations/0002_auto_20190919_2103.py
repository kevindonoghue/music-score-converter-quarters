# Generated by Django 2.2.4 on 2019-09-20 04:03

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='uploadedfile',
            old_name='key_signature',
            new_name='key',
        ),
    ]