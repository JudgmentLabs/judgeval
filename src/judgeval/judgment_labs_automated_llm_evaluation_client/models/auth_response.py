from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="AuthResponse")


@_attrs_define
class AuthResponse:
    """
    Attributes:
        api_key (str):
        organization_id (str):
        access_token (str):
        refresh_token (str):
    """

    api_key: str
    organization_id: str
    access_token: str
    refresh_token: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        api_key = self.api_key

        organization_id = self.organization_id

        access_token = self.access_token

        refresh_token = self.refresh_token

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "api_key": api_key,
                "organization_id": organization_id,
                "access_token": access_token,
                "refresh_token": refresh_token,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        api_key = d.pop("api_key")

        organization_id = d.pop("organization_id")

        access_token = d.pop("access_token")

        refresh_token = d.pop("refresh_token")

        auth_response = cls(
            api_key=api_key,
            organization_id=organization_id,
            access_token=access_token,
            refresh_token=refresh_token,
        )

        auth_response.additional_properties = d
        return auth_response

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
