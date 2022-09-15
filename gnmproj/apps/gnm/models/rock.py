from django.db import models
from django.utils.translation import ugettext_lazy as _

from .deposit import Deposit


class Rock(models.Model):
    class Meta:
        verbose_name = _('rock')
        verbose_name_plural = _('rocks')
        constraints = [
            models.UniqueConstraint(
                fields=['deposit', 'index'],
                name='unique deposit and index'
            )
        ]

    deposit = models.ForeignKey(
        Deposit,
        models.CASCADE,
        related_name='rocks',
        null=False,
        editable=True,
        verbose_name=_('deposit')
    )
    index = models.IntegerField(
        null=False,
        blank=False,
        editable=True,
        default=0,
        verbose_name=_('index')
    )
    name = models.CharField(
        max_length=63,
        null=False,
        blank=False,
        editable=True,
        verbose_name=_('name'),
    )
    color = models.CharField(
        null=False,
        blank=True,
        max_length=7,
        editable=True,
        default='',
        verbose_name=_('color'),
    )

    objects = models.Manager()

    def __str__(self):
        return f'{self.name}({self.deposit})'

    def save(self, *args, **kwargs):
        # index incrementer — credit to `tinfoilboy` @ https://stackoverflow.com/a/41230517
        if self._state.adding:
            last_id = self.__class__.objects.all().aggregate(largest=models.Max('index'))['largest']
            if last_id is not None:
                self.index = last_id + 1

        super(Rock, self).save(*args, **kwargs)
