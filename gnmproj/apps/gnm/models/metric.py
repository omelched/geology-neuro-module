from django.db import models
from django.utils.translation import ugettext_lazy as _


class Metric(models.Model):
    class Meta:
        verbose_name = _('metric')
        verbose_name_plural = _('metrics')
        constraints = []

    class MetricTypeEnum(models.TextChoices):
        CLASS = 'c', _('class stat')
        OVERALL = 'o', _('overall stat')

    name = models.CharField(
        max_length=63,
        null=False,
        blank=False,
        unique=True,
        editable=True,
        verbose_name=_('name'),
    )
    description = models.TextField(
        null=False,
        blank=True,
        default='',
        editable=True,
        verbose_name=_('description')
    )
    mtype = models.CharField(
        null=False,
        blank=False,
        max_length=1,
        choices=MetricTypeEnum.choices,
        editable=True,
        verbose_name=_('metric type')
    )

    objects = models.Manager()

    def __str__(self):
        return f'{self.name}'
