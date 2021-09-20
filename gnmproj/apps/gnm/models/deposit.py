from django.contrib.auth import get_user_model
from django.db import models
from django.utils.translation import ugettext_lazy as _


class Deposit(models.Model):
    class Meta:
        verbose_name = _('deposit')
        verbose_name_plural = _('deposits')

    name = models.CharField(
        max_length=63,
        null=False,
        editable=True,
        verbose_name=_('name'),
        unique=True
    )
    owners = models.ManyToManyField(
        get_user_model(),
        related_name='deposits',
        verbose_name=_('owners'),
    )
    bias = models.FloatField(
        null=False,
        blank=False,
        default=0,
        editable=True,
        verbose_name=_('bias'),
    )

    objects = models.Manager()

    def __str__(self):
        return f'{self.name}'


class DepositBorders(models.Model):
    class Meta:
        verbose_name = _('deposit border')
        verbose_name_plural = _('deposit borders')
        constraints = [
            models.UniqueConstraint(
                fields=['deposit', 'point_type'],
                name='unique deposit and point_type'
            )
        ]

    class PointTypeEnum(models.TextChoices):
        MIN = 'min', _('minimum')
        MAX = 'max', _('maximum')

    deposit = models.ForeignKey(
        Deposit,
        models.CASCADE,
        related_name='borders',
        null=False,
        editable=True,
        verbose_name=_('deposit')
    )
    point_type = models.CharField(
        null=False,
        blank=False,
        max_length=3,
        choices=PointTypeEnum.choices,
        editable=True,
        verbose_name=_('point type')
    )
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

    objects = models.Manager()

    def __str__(self):
        return f'{self.deposit}({self.point_type}) [{self.x:.2f}, {self.y:.2f}, {self.z:.2f}]'
