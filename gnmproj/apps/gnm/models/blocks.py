from django.db import models
from django.utils.translation import ugettext_lazy as _

from .well import Well
from .rock import Rock
from .neural_model import NeuralModel


class Block(models.Model):
    class Meta:
        abstract = True

    x = models.FloatField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('x'),
    )
    y = models.FloatField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('y'),
    )
    z = models.FloatField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('z'),
    )
    content = models.ForeignKey(
        Rock,
        models.CASCADE,
        related_name='+',
        null=False,
        editable=True,
        verbose_name=_('content')
    )


class KnownBlock(Block):
    class Meta:
        verbose_name = _('known block')
        verbose_name_plural = _('known blocks')
        constraints = [
            models.UniqueConstraint(
                fields=['well', 'size', 'x', 'y', 'z'],
                name='unique well, size, x, y, z'
            )
        ]

    well = models.ForeignKey(
        Well,
        models.CASCADE,
        related_name='known_blocks',
        null=False,
        editable=True,
        verbose_name=_('well')
    )
    size = models.FloatField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('size'),
    )

    objects = models.Manager()

    def __str__(self):
        return f'{self.well} (size: {self.size:.2f}, ({self.x:.2f}, {self.y:.2f}, {self.z:.2f}))'


class PredictedBlock(Block):
    class Meta:
        verbose_name = _('predicted block')
        verbose_name_plural = _('predicted blocks')
        constraints = [
            models.UniqueConstraint(
                fields=['neural_model', 'x', 'y', 'z'],
                name='unique neural_model, x, y, z'
            )
        ]

    neural_model = models.ForeignKey(
        NeuralModel,
        models.CASCADE,
        related_name='predicted_blocks',
        null=False,
        editable=True,
        verbose_name=_('neural_model')
    )
    known_block = models.ForeignKey(
        KnownBlock,
        models.CASCADE,
        related_name='predicted_blocks',
        null=True,
        editable=True,
        verbose_name=_('known_block')
    )

    objects = models.Manager()

    def __str__(self):
        return f'{self.neural_model} ({self.x:.2f}, {self.y:.2f}, {self.z:.2f})'


class PredictedBlockOutputs(models.Model):
    class Meta:
        verbose_name = _('predicted block output')
        verbose_name_plural = _('predicted block outputs')
        constraints = []

    predicted_block = models.ForeignKey(
        PredictedBlock,
        models.CASCADE,
        related_name='outputs',
        null=False,
        editable=True,
        verbose_name=_('predicted_block')
    )
    rock = models.ForeignKey(
        Rock,
        models.CASCADE,
        related_name='+',
        null=True,
        editable=True,
        verbose_name=_('rock')
    )
    value = models.FloatField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('value'),
    )

    objects = models.Manager()

    def __str__(self):
        return f'{self.predicted_block}/{self.rock}[{self.value:.2f}]'
