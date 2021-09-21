from django.db import models
from django.utils.translation import ugettext_lazy as _

from .deposit import Deposit
from .rock import Rock


class Well(models.Model):
    class Meta:
        verbose_name = _('well')
        verbose_name_plural = _('wells')
        constraints = []

    deposit = models.ForeignKey(
        Deposit,
        models.CASCADE,
        related_name='wells',
        null=False,
        verbose_name=_('deposit')
    )
    head_x = models.FloatField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('head_x'),
    )
    head_y = models.FloatField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('head_y'),
    )
    head_z = models.FloatField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('head_z'),
    )
    tail_x = models.FloatField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('tail_x'),
    )
    tail_y = models.FloatField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('tail_y'),
    )
    tail_z = models.FloatField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('tail_z'),
    )

    objects = models.Manager()

    def __str__(self):
        return f'{self.deposit} [({self.head_x:.2f}, {self.head_y:.2f}, {self.head_z:.2f}),' \
               f' ({self.tail_x:.2f}, {self.tail_y:.2f}, {self.tail_z:.2f})]'


class WellInterval(models.Model):
    class Meta:
        verbose_name = _('well interval')
        verbose_name_plural = _('well intervals')
        constraints = [
            models.UniqueConstraint(
                fields=['well', 'position'],
                name='unique well and position'
            )
        ]

    well = models.ForeignKey(
        Well,
        models.CASCADE,
        related_name='intervals',
        null=False,
        verbose_name=_('well')
    )
    position = models.IntegerField(
        null=False,
        blank=False,
        editable=True,
        default=0,
        verbose_name=_('position')
    )
    from_x = models.FloatField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('from_x'),
    )
    from_y = models.FloatField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('from_y'),
    )
    from_z = models.FloatField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('from_z'),
    )
    to_x = models.FloatField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('to_x'),
    )
    to_y = models.FloatField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('to_y'),
    )
    to_z = models.FloatField(
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('to_z'),
    )
    depth_to = models.FloatField(
        null=False,
        blank=False,
        default=.0,
        editable=True,
        verbose_name=_('depth_to'),
    )
    depth_from = models.FloatField(
        null=False,
        blank=False,
        default=.0,
        editable=True,
        verbose_name=_('depth_from'),
    )
    zenit = models.FloatField(
        null=False,
        blank=False,
        default=.0,
        editable=True,
        verbose_name=_('zenit'),
    )
    azimut = models.FloatField(
        null=False,
        blank=False,
        default=.0,
        editable=True,
        verbose_name=_('azimut'),
    )
    content = models.ForeignKey(
        Rock,
        models.CASCADE,
        related_name='+',
        null=False,
        editable=True,
        verbose_name=_('content')
    )

    objects = models.Manager()

    def __str__(self):
        return f'{self.well} / [{self.position}]'

    def save(self, *args, **kwargs):
        # index incrementer — credit to `tinfoilboy` @ https://stackoverflow.com/a/41230517
        if self._state.adding:
            last_id = self.objects.all().aggregate(largest=models.Max('position'))['largest']
            if last_id is not None:
                self.position = last_id + 1

        super(WellInterval, self).save(*args, **kwargs)
