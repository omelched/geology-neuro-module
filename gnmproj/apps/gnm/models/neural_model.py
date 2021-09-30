from django.db import models
from django.utils.translation import ugettext_lazy as _

from .deposit import Deposit
from .well import Well
from .metric import Metric
from .rock import Rock


class CrossValidation(models.Model):
    class Meta:
        verbose_name = _('cross-validation')
        verbose_name_plural = _('cross-validations')
        constraints = []

    name = models.CharField(
        max_length=63,
        null=False,
        blank=True,
        default='',
        editable=True,
        verbose_name=_('name'),
    )

    def __str__(self):
        return self.name or self.id


class NeuralModel(models.Model):
    class Meta:
        verbose_name = _('neural model')
        verbose_name_plural = _('neural models')
        constraints = []

    deposit = models.ForeignKey(
        Deposit,
        models.CASCADE,
        related_name='neural_models',
        null=False,
        editable=True,
        verbose_name=_('deposit')
    )
    block_size = models.FloatField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('block_size'),
    )
    max_epochs = models.IntegerField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('max_epochs')
    )
    cross_validation = models.ForeignKey(
        CrossValidation,
        models.CASCADE,
        related_name='neural_models',
        null=True,
        editable=True,
        verbose_name=_('cross-validation')
    )
    dump = models.BinaryField(
        null=True,
        blank=False
    )
    excluded_wells = models.ManyToManyField(
        Well,
        related_name='excluded_in',
        verbose_name=_('excluded wells'),
    )

    objects = models.Manager()

    def __str__(self):
        return f'{self.deposit} (size: {self.block_size}, epochs: {self.max_epochs})' \
               + (' [CV]' if self.cross_validation else '')


class NeuralModelMetricValues(models.Model):
    class Meta:
        verbose_name = _('neural model metric value')
        verbose_name_plural = _('neural model metric values')
        constraints = [
            models.UniqueConstraint(
                fields=['neural_model', 'metric', 'epoch', 'rock'],
                name='unique meural_model, metric epoch, rock'
            )
        ]

    neural_model = models.ForeignKey(
        NeuralModel,
        models.CASCADE,
        related_name='metric_values',
        null=False,
        editable=True,
        verbose_name=_('neural model')
    )
    metric = models.ForeignKey(
        Metric,
        models.CASCADE,
        related_name='values',
        null=False,
        editable=True,
        verbose_name=_('metric')
    )
    epoch = models.IntegerField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('epoch')
    )
    rock = models.ForeignKey(
        Rock,
        models.CASCADE,
        related_name='class_stat_values',
        null=True,
        editable=True,
        verbose_name=_('rock')
    )
    value = models.TextField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('value')
    )

    objects = models.Manager()

    def __str__(self):
        return f'{self.neural_model} / {self.metric} [{self.epoch}]' \
               + (f' {self.rock.name}' if self.rock else '')
